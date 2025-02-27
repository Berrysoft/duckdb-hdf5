use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    ffi,
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab},
    Connection, Result,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use hdf5::types::{
    CompoundType, FloatSize, IntSize, TypeDescriptor, VarLenArray, VarLenAscii, VarLenUnicode,
};
use std::{
    borrow::Cow,
    error::Error,
    ops::Deref,
    sync::atomic::{AtomicUsize, Ordering},
};

pub trait ReadRawBytes {
    fn read_raw_bytes(&self, dtype: &TypeDescriptor) -> hdf5::Result<Vec<u8>>;
}

impl ReadRawBytes for hdf5::Dataset {
    fn read_raw_bytes(&self, dtype: &TypeDescriptor) -> hdf5::Result<Vec<u8>> {
        let len = self.size();
        let item_size = dtype.size();
        let mut buffer = Vec::with_capacity(len * item_size);
        // Convert again to fit the current native endian.
        let native_dtype = hdf5::Datatype::from_descriptor(dtype)?;
        hdf5::h5call!(hdf5_sys::h5d::H5Dread(
            self.id(),
            native_dtype.id(),
            hdf5_sys::h5s::H5S_ALL,
            hdf5_sys::h5s::H5S_ALL,
            hdf5_sys::h5p::H5P_DEFAULT,
            buffer.spare_capacity_mut().as_mut_ptr() as *mut _
        ))?;
        unsafe {
            buffer.set_len(len * item_size);
        }
        Ok(buffer)
    }
}

struct Hdf5ReadBindData {
    dtype: TypeDescriptor,
    data: Vec<u8>,
}

const RESULT_COLNAME: Cow<str> = Cow::Borrowed("result");

fn iter_dtype(dtype: &TypeDescriptor) -> Vec<(Cow<'static, str>, LogicalTypeHandle)> {
    match dtype {
        TypeDescriptor::Integer(size) => {
            let ty = match size {
                IntSize::U1 => LogicalTypeId::Tinyint,
                IntSize::U2 => LogicalTypeId::Smallint,
                IntSize::U4 => LogicalTypeId::Integer,
                IntSize::U8 => LogicalTypeId::Bigint,
            };
            vec![(RESULT_COLNAME, ty.into())]
        }
        TypeDescriptor::Unsigned(size) => {
            let ty = match size {
                IntSize::U1 => LogicalTypeId::UTinyint,
                IntSize::U2 => LogicalTypeId::USmallint,
                IntSize::U4 => LogicalTypeId::UInteger,
                IntSize::U8 => LogicalTypeId::UBigint,
            };
            vec![(RESULT_COLNAME, ty.into())]
        }
        TypeDescriptor::Float(size) => {
            let ty = match size {
                FloatSize::U4 => LogicalTypeId::Float,
                FloatSize::U8 => LogicalTypeId::Double,
            };
            vec![(RESULT_COLNAME, ty.into())]
        }
        TypeDescriptor::Boolean => vec![(RESULT_COLNAME, LogicalTypeId::Boolean.into())],
        TypeDescriptor::Enum(e) => iter_dtype(&e.base_type()),
        TypeDescriptor::Compound(c) => {
            let mut res = vec![];
            for f in &c.fields {
                let (names, types) = iter_dtype(&f.ty)
                    .into_iter()
                    .unzip::<Cow<'static, str>, LogicalTypeHandle, Vec<_>, Vec<_>>();
                let ty = if types.len() > 1 {
                    let types = types
                        .into_iter()
                        .zip(&names)
                        .map(|(ty, name)| (name.deref(), ty))
                        .collect::<Vec<_>>();
                    LogicalTypeHandle::struct_type(types.as_slice())
                } else {
                    types.into_iter().next().unwrap()
                };
                res.push((Cow::Owned(f.name.clone()), ty))
            }
            res
        }
        TypeDescriptor::FixedArray(ty, len) => {
            let inner = &iter_dtype(ty)[0].1;
            let ty = LogicalTypeHandle::array(inner, *len as _);
            vec![(RESULT_COLNAME, ty)]
        }
        TypeDescriptor::VarLenArray(ty) => {
            let inner = &iter_dtype(ty)[0].1;
            let ty = LogicalTypeHandle::list(inner);
            vec![(RESULT_COLNAME, ty)]
        }
        TypeDescriptor::FixedAscii(_)
        | TypeDescriptor::FixedUnicode(_)
        | TypeDescriptor::VarLenAscii
        | TypeDescriptor::VarLenUnicode => {
            vec![(RESULT_COLNAME, LogicalTypeId::Varchar.into())]
        }
        TypeDescriptor::Reference(_) => vec![(RESULT_COLNAME, LogicalTypeId::Blob.into())],
    }
}

macro_rules! fill_vec {
    ($output:expr, $idx:expr, $slice:expr, $t:ty) => {{
        let mut vec = $output.flat_vector($idx);
        vec.as_mut_slice::<$t>()[0] = unsafe { $slice.as_ptr().cast::<$t>().read_unaligned() };
    }};
}

fn fill(dtype: &TypeDescriptor, slice: &[u8], output: &mut DataChunkHandle, idx: usize) {
    match dtype {
        TypeDescriptor::Integer(IntSize::U1) => fill_vec!(output, idx, slice, i8),
        TypeDescriptor::Integer(IntSize::U2) => fill_vec!(output, idx, slice, i16),
        TypeDescriptor::Integer(IntSize::U4) => fill_vec!(output, idx, slice, i32),
        TypeDescriptor::Integer(IntSize::U8) => fill_vec!(output, idx, slice, i64),
        TypeDescriptor::Unsigned(IntSize::U1) => fill_vec!(output, idx, slice, u8),
        TypeDescriptor::Unsigned(IntSize::U2) => fill_vec!(output, idx, slice, u16),
        TypeDescriptor::Unsigned(IntSize::U4) => fill_vec!(output, idx, slice, u32),
        TypeDescriptor::Unsigned(IntSize::U8) => fill_vec!(output, idx, slice, u64),
        TypeDescriptor::Float(FloatSize::U4) => fill_vec!(output, idx, slice, f32),
        TypeDescriptor::Float(FloatSize::U8) => fill_vec!(output, idx, slice, f64),
        TypeDescriptor::Boolean => fill_vec!(output, idx, slice, bool),
        TypeDescriptor::Enum(e) => fill(&e.base_type(), slice, output, idx),
        TypeDescriptor::Compound(c) => {
            for (i, f) in c.fields.iter().enumerate() {
                fill(&f.ty, &slice[f.offset..], output, idx + i);
            }
        }
        TypeDescriptor::FixedArray(ty, len) => {
            let vec = output.array_vector(idx);
            vec.set_child(&slice[..*len * ty.size()]);
        }
        TypeDescriptor::VarLenArray(ty) => {
            let array = unsafe { slice.as_ptr().cast::<VarLenArray<u8>>().as_ref() }.unwrap();
            let vec = output.list_vector(idx);
            vec.set_child(&slice[..array.len() * ty.size()]);
        }
        TypeDescriptor::FixedAscii(len) | TypeDescriptor::FixedUnicode(len) => {
            let vec = output.array_vector(idx);
            vec.set_child(&slice[..*len]);
        }
        TypeDescriptor::VarLenAscii => {
            let array = unsafe { slice.as_ptr().cast::<VarLenAscii>().as_ref() }.unwrap();
            let vec = output.list_vector(idx);
            vec.set_child(array.as_bytes());
        }
        TypeDescriptor::VarLenUnicode => {
            let array = unsafe { slice.as_ptr().cast::<VarLenUnicode>().as_ref() }.unwrap();
            let vec = output.list_vector(idx);
            vec.set_child(array.as_bytes());
        }
        TypeDescriptor::Reference(_) => {
            let vec = output.flat_vector(idx);
            vec.insert(0, &slice[..dtype.size()]);
        }
    }
}

impl Hdf5ReadBindData {
    fn new(path: &str, dataset: &str) -> hdf5::Result<Self> {
        let file = hdf5::File::open(path)?;
        let dataset = file.dataset(dataset)?;
        let dtype = dataset.dtype()?.to_descriptor()?;
        let data = dataset.read_raw_bytes(&dtype)?;
        Ok(Self { dtype, data })
    }

    fn iter_dtype(&self) -> Vec<(Cow<'static, str>, LogicalTypeHandle)> {
        iter_dtype(&self.dtype)
    }

    fn project_dtype(&self, indices: &[duckdb::ffi::idx_t]) -> TypeDescriptor {
        match &self.dtype {
            TypeDescriptor::Compound(c) => {
                let mut fields = vec![];
                for i in indices {
                    fields.push(c.fields[*i as usize].clone());
                }
                TypeDescriptor::Compound(CompoundType {
                    fields,
                    size: c.size,
                })
            }
            _ => self.dtype.clone(),
        }
    }

    fn fill(&self, index: usize, dtype: &TypeDescriptor, output: &mut DataChunkHandle) {
        let item_size = self.dtype.size();
        if index * item_size >= self.data.len() {
            output.set_len(0);
        } else {
            let data = &self.data[index * item_size..][..item_size];
            fill(dtype, data, output, 0);
            output.set_len(1);
        }
    }
}

struct Hdf5ReadInitData {
    index: AtomicUsize,
    dtype: TypeDescriptor,
}

impl Hdf5ReadInitData {
    pub fn new(dtype: TypeDescriptor) -> Self {
        Self {
            index: AtomicUsize::new(0),
            dtype,
        }
    }
}

struct Hdf5Read;

impl VTab for Hdf5Read {
    type InitData = Hdf5ReadInitData;
    type BindData = Hdf5ReadBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let path = bind.get_parameter(0).to_string();
        let dataset = bind.get_parameter(1).to_string();
        let data = Hdf5ReadBindData::new(&path, &dataset)?;
        for (name, dtype) in data.iter_dtype() {
            bind.add_result_column(&name, dtype);
        }
        Ok(data)
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        let bind_data = unsafe { init.get_bind_data::<Self::BindData>().as_ref() }.unwrap();
        let dtype = bind_data.project_dtype(&init.get_column_indices());
        Ok(Hdf5ReadInitData::new(dtype))
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let bind_data = func.get_bind_data();
        let init_data = func.get_init_data();
        let index = init_data.index.fetch_add(1, Ordering::Relaxed);
        bind_data.fill(index, &init_data.dtype, output);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeId::Varchar.into(),
            LogicalTypeId::Varchar.into(),
        ])
    }

    fn supports_pushdown() -> bool {
        true
    }
}

#[duckdb_entrypoint_c_api()]
pub fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<Hdf5Read>("read_hdf5")?;
    Ok(())
}
