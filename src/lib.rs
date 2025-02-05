use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    ffi,
    vtab::{BindInfo, Free, FunctionInfo, InitInfo, VTab},
    Connection, Result,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use std::error::Error;

struct Hdf5ReadBindData {
    path: Option<String>,
    dataset: Option<String>,
}

impl Free for Hdf5ReadBindData {
    fn free(&mut self) {
        self.path.take();
        self.dataset.take();
    }
}

struct Hdf5ReadInitData {
    done: bool,
}

impl Free for Hdf5ReadInitData {}

struct Hdf5Read;

impl VTab for Hdf5Read {
    type InitData = Hdf5ReadInitData;
    type BindData = Hdf5ReadBindData;

    unsafe fn bind(bind: &BindInfo, data: *mut Self::BindData) -> Result<(), Box<dyn Error>> {
        bind.add_result_column("path", LogicalTypeId::Varchar.into());
        bind.add_result_column("dataset", LogicalTypeId::Varchar.into());
        let path = bind.get_parameter(0).to_string();
        let dataset = bind.get_parameter(1).to_string();
        if let Some(data) = data.as_mut() {
            data.path = Some(path);
            data.dataset = Some(dataset);
        }
        Ok(())
    }

    unsafe fn init(_: &InitInfo, data: *mut Self::InitData) -> Result<(), Box<dyn Error>> {
        if let Some(data) = data.as_mut() {
            data.done = false;
        }
        Ok(())
    }

    unsafe fn func(
        func: &FunctionInfo,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init_info = func.get_init_data::<Self::InitData>().as_mut();
        let bind_info = func.get_bind_data::<Self::BindData>().as_mut();

        if let Some(init_info) = init_info {
            if init_info.done {
                output.set_len(0);
            } else {
                init_info.done = true;
                let path_vec = output.flat_vector(0);
                let path = bind_info
                    .as_ref()
                    .and_then(|bind_info| bind_info.path.as_deref())
                    .unwrap_or_default();
                path_vec.insert(0, path);
                let dataset_vec = output.flat_vector(1);
                let dataset = bind_info
                    .as_ref()
                    .and_then(|bind_info| bind_info.dataset.as_deref())
                    .unwrap_or_default();
                dataset_vec.insert(0, dataset);
                output.set_len(1);
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
