use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    ffi,
    vtab::{BindInfo, Free, FunctionInfo, InitInfo, VTab},
    Connection, Result,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use std::{borrow::Cow, error::Error};

struct BindDataInner {
    file: hdf5::File,
    dataset: hdf5::Dataset,
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

impl BindDataInner {
    fn new(path: &str, dataset: &str) -> hdf5::Result<Self> {
        let file = hdf5::File::open(path)?;
        let dataset = file.dataset(dataset)?;
        Ok(Self { file, dataset })
    }

    fn iter_dtype(&self) -> hdf5::Result<Vec<(Cow<'static, str>, LogicalTypeHandle)>> {
        let dtype = self.dataset.dtype()?.to_descriptor()?;
        Ok(iter_dtype(&dtype))
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
            for (name, dtype) in inner.iter_dtype()? {
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
            if let Some(data) = bind_info.data.take() {
                todo!()
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
