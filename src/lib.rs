use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    ffi,
    vtab::{BindInfo, Free, FunctionInfo, InitInfo, VTab},
    Connection, Result,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use std::{borrow::Cow, error::Error};

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

struct BindDataInner {
    dtype: TypeDescriptor,
    data: Vec<u8>,
    index: usize,
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
                let ty = iter_dtype(&f.ty).into_iter().next().unwrap().1;
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
        TypeDescriptor::Reference(_) => todo!(),
    }
}

macro_rules! fill_vec {
    ($output:expr, $idx:expr, $slice:expr, $t:ty) => {{
        let mut vec = $output.flat_vector($idx);
        vec.as_mut_slice::<$t>()[0] = unsafe { $slice.as_ptr().cast::<$t>().read_unaligned() };
    }};
}

fn fill(dtype: &TypeDescriptor, slice: &[u8], output: &mut DataChunkHandle) {
    match dtype {
        TypeDescriptor::Integer(IntSize::U1) => fill_vec!(output, 0, slice, i8),
        TypeDescriptor::Integer(IntSize::U2) => fill_vec!(output, 0, slice, i16),
        TypeDescriptor::Integer(IntSize::U4) => fill_vec!(output, 0, slice, i32),
        TypeDescriptor::Integer(IntSize::U8) => fill_vec!(output, 0, slice, i64),
        TypeDescriptor::Unsigned(IntSize::U1) => fill_vec!(output, 0, slice, u8),
        TypeDescriptor::Unsigned(IntSize::U2) => fill_vec!(output, 0, slice, u16),
        TypeDescriptor::Unsigned(IntSize::U4) => fill_vec!(output, 0, slice, u32),
        TypeDescriptor::Unsigned(IntSize::U8) => fill_vec!(output, 0, slice, u64),
        TypeDescriptor::Float(FloatSize::U4) => fill_vec!(output, 0, slice, f32),
        TypeDescriptor::Float(FloatSize::U8) => fill_vec!(output, 0, slice, f64),
        TypeDescriptor::Boolean => fill_vec!(output, 0, slice, bool),
        TypeDescriptor::Enum(e) => fill(&e.base_type(), slice, output),
        TypeDescriptor::Compound(c) => {
            todo!()
        }
        TypeDescriptor::FixedArray(ty, len) => {
            todo!()
        }
        TypeDescriptor::VarLenArray(ty) => {
            todo!()
        }
        TypeDescriptor::FixedAscii(len) | TypeDescriptor::FixedUnicode(len) => {
            todo!()
        }
        TypeDescriptor::VarLenAscii | TypeDescriptor::VarLenUnicode => {
            todo!()
        }
        TypeDescriptor::Reference(_) => todo!(),
    }
}

impl BindDataInner {
    fn new(path: &str, dataset: &str) -> hdf5::Result<Self> {
        let file = hdf5::File::open(path)?;
        let dataset = file.dataset(dataset)?;
        let dtype = dataset.dtype()?.to_descriptor()?;
        let data = dataset.read_raw_bytes(&dtype)?;
        Ok(Self {
            dtype,
            data,
            index: 0,
        })
    }

    fn iter_dtype(&self) -> Vec<(Cow<'static, str>, LogicalTypeHandle)> {
        iter_dtype(&self.dtype)
    }

    fn fill(&mut self, output: &mut DataChunkHandle) -> bool {
        let item_size = self.dtype.size();
        if self.index * item_size >= self.data.len() {
            output.set_len(0);
            return false;
        }
        let data = &self.data[self.index * item_size..][..item_size];
        self.index += 1;
        fill(&self.dtype, data, output);
        output.set_len(1);
        true
    }
}

struct Hdf5ReadBindData {
    data: Option<BindDataInner>,
}

impl Free for Hdf5ReadBindData {
    fn free(&mut self) {
        self.data.take();
    }
}

struct Hdf5ReadInitData;

impl Free for Hdf5ReadInitData {}

struct Hdf5Read;

impl VTab for Hdf5Read {
    type InitData = Hdf5ReadInitData;
    type BindData = Hdf5ReadBindData;

    unsafe fn bind(bind: &BindInfo, data: *mut Self::BindData) -> Result<(), Box<dyn Error>> {
        let path = bind.get_parameter(0).to_string();
        let dataset = bind.get_parameter(1).to_string();
        if let Some(data) = data.as_mut() {
            let inner = BindDataInner::new(&path, &dataset)?;
            for (name, dtype) in inner.iter_dtype() {
                bind.add_result_column(&name, dtype);
            }
            data.data = Some(inner);
        }
        Ok(())
    }

    unsafe fn init(_: &InitInfo, _data: *mut Self::InitData) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    unsafe fn func(
        func: &FunctionInfo,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let bind_info = func.get_bind_data::<Self::BindData>().as_mut();

        if let Some(bind_info) = bind_info {
            if let Some(mut data) = bind_info.data.take() {
                if data.fill(output) {
                    bind_info.data = Some(data);
                }
            } else {
                output.set_len(0);
            }
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeId::Varchar.into(),
            LogicalTypeId::Varchar.into(),
        ])
    }
}

#[duckdb_entrypoint_c_api(ext_name = "hdf5", min_duckdb_version = "v0.0.1")]
pub fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<Hdf5Read>("hdf5_read")?;
    Ok(())
}
