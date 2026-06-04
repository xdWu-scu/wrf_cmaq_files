#make monthly list for CMAQ
import os
from multiprocessing import Pool
import glob
import os.path
import tqdm
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import re
from osgeo import gdal, osr
import pandas as pd
import PseudoNetCDF as pnc
import datetime
import netCDF4 as nc
import numpy as np
from tqdm import tqdm

def getNearestPos(station_lat, station_lon, XLAT, XLONG):
    """
    得到距离站点最近的经纬度索引值
    :param station_lat:
    :param station_lon:
    :param XLAT:
    :param XLONG:
    :return:
    """
    difflat = station_lat - XLAT  # 经纬度数组与站点经纬度相减，找出最小的
    difflon = station_lon - XLONG
    rad = np.multiply(difflat, difflat) + np.multiply(difflon, difflon)  # difflat * difflat + difflon * difflon 计算最小的距离
    aa = np.where(rad == np.min(rad))  # 查询最小的距离的点在哪里，也就是站点位置
    ind = np.squeeze(np.array(aa))

    return ind

def getMultiNearestPos(station_coords, XLAT, XLONG,num_processes):
    """
    对多个站点并行找到最近的经纬度索引值
    :param station_coords: 包含多个站点的经纬度的列表 [(lat1, lon1), (lat2, lon2), ...]
    :param XLAT: 纬度数组
    :param XLONG: 经度数组
    :return: 最近经纬度的索引列表
    """
    # 创建参数列表
    args = [(lat, lon, XLAT, XLONG) for lat, lon in station_coords]

    with Pool(processes=num_processes) as pool:
        results = pool.map(getNearestPos, args)

    return results

def tiff2ArrayWithLatLon(tif_dir):
    """
    读取一个tiff，得到它的数据以及经纬度数组，并返回像素行列长宽(分辨率)
    :param tif_dir:
    :return:
    """
    with rasterio.open(tif_dir) as src:
        # 读取数据数组
        data_array = src.read(1)  # 读取第一个波段的数据
        # 获取仿射变换参数
        transform = src.transform

        p_x = transform[0]  #
        p_y = -transform[4]

        # 获取经纬度网格点
        height, width = data_array.shape
        lon, lat = np.meshgrid(
            np.arange(width) * transform[0] + transform[2],  # 经度
            np.arange(height) * transform[4] + transform[5]  # 纬度
        )

    return [data_array, lat, lon, p_x, p_y]
def getLatLonAreaFormTiff(
        input_tif,
        latmax, lonmin, latmin, lonmax,
        disable_tqdm=False
):
    """
    根据tiff的信息，先获取其对应行列号的经纬度及其数据，
    然后计算对应纬度范围内的数据所在的array行列范围，同时范围对应的经纬度数组
    :return:
    """
    with rasterio.open(input_tif) as src:
        # 读取数据数组
        data_array = src.read(1)  # 读取第一个波段的数据
        # 获取仿射变换参数
        transform = src.transform

        # 获取经纬度网格点
        height, width = data_array.shape
        LON, LAT = np.meshgrid(
            np.arange(width) * transform[0] + transform[2],  # 经度
            np.arange(height) * transform[4] + transform[5]  # 纬度
        )
    # print(LAT,LON)
    # 根据geoem四川盆地左上和右下点，筛选出四川盆地内的部分
    leftup = getNearestPos(latmax, lonmin, LAT, LON)  # 从WRF得到站点格子
    rightdown = getNearestPos(latmin, lonmax, LAT, LON)  # 从WRF得到站点格子

    data = data_array[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]
    LAT = LAT[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]
    LON = LON[leftup[0]:rightdown[0], leftup[1]:rightdown[1]]

    return [data, LAT, LON]
#nc to tif
def MEIC2Geotiff(input_dir,output_dir):
    print("MEIC2Geotiff: This script is written by Jiaxin Qiu.")
   
    if os.path.exists(output_dir) is False:
         os.mkdir(output_dir)
 
    files = glob.glob(f"{input_dir}/*.nc")
 
    for file in tqdm(files):
        sub_name = os.path.basename(file)
        condition = f"(.*?)_(.*?)_(.*?)_(.*?).nc"
        # condition = f"(.*?)_(.*?)__(.*?)__(.*?).asc"  # For 2019 and 2020.
        encode_name = re.findall(condition, sub_name)[0]
        year = r"%.4d" % int(encode_name[0])
        mm = r"%.2d" % int(encode_name[1])
        sector = encode_name[2]
        pollutant = encode_name[3].replace(".", "")
        output_name = f"{output_dir}/MEIC_{year}_{mm}__{sector}__{pollutant}.tiff"
 
        lines = xr.open_dataset(file)
        # 打印读取到的数组（列表）
        _ = np.array(lines.variables['z'][:])
        z = np.where(_ == -9999.0, 0.0, _)
        
        # 读取nc的行和列
        width = np.array(lines.variables['dimension'][0])
        height = np.array(lines.variables['dimension'][1])
        z_2d = z.reshape(height, width)
        
        
        # 最大最小经纬度
        min_long, min_lat, max_long, max_lat = 70.0, 10.0, 150.0, 60.0
        
        # 分辨率
        x_resolution = 0.25
        y_resolution = 0.25
        
        
        
        # 创建GeoTIFF文件的变换矩阵
        transform = from_bounds(min_long, min_lat, max_long, max_lat, width, height)
        
        # 定义GeoTIFF文件的元数据
        metadata = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        }
        
        # 创建GeoTIFF文件
        with rasterio.open(output_name, "w", **metadata) as dst:
            dst.write(z_2d,1)  # 将数据写入波段1

sectors = [ 'transportation', 'residential', 'power', 'agriculture','industry']#
Workdir = os.getcwd()
# PseudoNetCDF打开griddesc文件，并生成对应的nc文件框架
# os.makedirs(Workdir, exist_ok=True)
target_mechanism = 'CB06'
grid_name = 'SAPRC_d01'
start_date = "2024-08-01"
end_date = "2024-09-01"
yyyymmn = '202408'
grid_desc = f'E:\\dif_chem_con\\MEIC_EMISSION_INPUT\\{grid_name}_{yyyymmn}\\GRIDDESC'
GRIDCRO2D_dir = f'E:\\dif_chem_con\\MEIC_EMISSION_INPUT\\{grid_name}_{yyyymmn}\\GRIDCRO2D_20240801.nc'



sectors = sectors  # 生成的排放源部门

# # 转化MEIC原始数据为tiff：
MEIC_dir = f'{Workdir}\\MEIC_{target_mechanism}\\'
MEIC_tiffs_dir = f'{Workdir}\\MEIC_tiffs\\'
os.makedirs(MEIC_tiffs_dir, exist_ok=True)
input_dir = MEIC_tiffs_dir
# MEIC2Geotiff(MEIC_dir,MEIC_tiffs_dir)


output_dir =f'{Workdir}\\{target_mechanism}_emission_{grid_name}_{yyyymmn}\\'
os.makedirs(output_dir, exist_ok=True)

# periods = pd.period_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq='D')
periods = pd.period_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq='D')
periods_H = pd.period_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq='H')
# 从GRIDCRO2D获取经纬度范围信息
GRIDCRO2D = nc.Dataset(GRIDCRO2D_dir)
LON = np.array(GRIDCRO2D.variables['LON'][:][0][0])
LAT = np.array(GRIDCRO2D.variables['LAT'][:][0][0])
GRID_p_x = LON[0][1] - LON[0][0]  # GRID的xy分辨率
GRID_p_y = LAT[1][0] - LAT[0][0]
print(GRID_p_x, GRID_p_y)
ROW = LON.shape[0]  # 以LON作为标准数据来获取数组行列
COL = LON.shape[1]
lonmin, latmax, lonmax, latmin = (LON.min(), LAT.max(),
                                  LON.max(), LAT.min())  # 获取griddesc区域矩形经纬度范围
data_Temp = np.zeros((ROW, COL), dtype=float, order="c")  # griddesc空间格点对应的空数组
special_col1 ={'AACD':1, 'FACD':1,'KET':1,'ACET':1,'PRPA':0.8,'ETHY':1,'BENZ':1,'CH4':1}
special_col2 ={'APIN':1}
data_allocators_interp = {}
for sector in sectors:
    
    _weekly_factor = pd.read_csv(r"E:\\dif_chem_con\\MEIC_EMISSION_INPUT\\split_factor\\weekly.csv")
    _hourly_factor = pd.read_csv(r"E:\\dif_chem_con\\MEIC_EMISSION_INPUT\\split_factor\\hourly.csv")
    weekly_factor = _weekly_factor[sector].values
    hourly_factor = _hourly_factor[sector].values
   
    #for date in periods:
    date =  start_date
    month = str(date).split('-')[1]
    w = datetime.datetime.strftime(pd.to_datetime(str(date)), "%w")  # 此日为星期几
    # hou = datetime.datetime.strftime(pd.to_datetime(str(date)), "%H")
    yyyymmdd = datetime.datetime.strftime(pd.to_datetime(str(date)), "%Y%m%d")
    yyyyjjj = datetime.datetime.strftime(pd.to_datetime(str(date)), "%Y%j")
    
    print(date,sector)
    
    gf = pnc.pncopen(
        grid_desc,
        GDNAM=grid_name,
        format="griddesc",
        SDATE=int(yyyyjjj),
        TSTEP=10000,
        withcf=False,
    )
    gf.updatetflag(overwrite=True)
    tmpf = gf.sliceDimensions(TSTEP=[0] * (len(periods_H)-1))
    max_col_index = getattr(tmpf, "NCOLS") - 1
    max_row_index = getattr(tmpf, "NROWS") - 1
    
    
    
    # Read species file.
    species_file = rf"{Workdir}//species//MEIC-CB05_CB06_speciate_{sector}.xlsx"
    species_info = pd.read_excel(species_file)
    # fname_list = species_info.pollutant.values
    var_list = species_info.pollutant.values
    # factor_list = species_info.split_factor.values
    divisor_list = species_info.divisor.values
    origin_units = species_info.inv_unit.values
    target_units = species_info.emi_unit.values
    
    for emission_specie in species_info['emission_species']:
    
        # emission_specie = 'NO2'
        colnum = list(species_info['emission_species']).index(emission_specie) # 找到列号，以获取物种其他信息
        species_name = species_info['pollutant'][colnum]
        if species_name=='POC':
           species_name = 'OC'
        elif species_name=='PEC':
           species_name = 'BC'
        else:
           species_name = species_name
           
        if emission_specie=='PMC':
            MEIC_tiff_dir1 = f"{MEIC_tiffs_dir}MEIC_2020_{month}__{sector}__PM10.tiff"
            data_MEIC_pre1 = getLatLonAreaFormTiff(MEIC_tiff_dir1,latmax, lonmin, latmin, lonmax)
            MEIC_tiff_dir2 = f"{MEIC_tiffs_dir}MEIC_2020_{month}__{sector}__PM25.tiff"
            data_MEIC_pre2 = getLatLonAreaFormTiff(MEIC_tiff_dir2, latmax, lonmin, latmin, lonmax)
            data_MEIC = data_MEIC_pre1[0] - data_MEIC_pre2[0]
            LAT_MEIC, LON_MEIC =  data_MEIC_pre1[1], data_MEIC_pre1[2]
        else:
            # 路径判断
            if emission_specie in special_col1.keys():
                MEIC_tiffs_dir = 'E:\\dif_chem_con\\MEIC_EMISSION_INPUT\\SAPRC07\\MEIC_tiffs\\'
                print(f'{emission_specie} was change path to SAPRC07')
                MEIC_tiff_dir = f"{MEIC_tiffs_dir}MEIC_2020_{month}__{sector}__{species_name}.tiff"
                data_MEIC_pre = getLatLonAreaFormTiff(MEIC_tiff_dir, latmax, lonmin, latmin, lonmax)
                data_MEIC, LAT_MEIC, LON_MEIC = data_MEIC_pre[0]*special_col1[emission_specie ], data_MEIC_pre[1], data_MEIC_pre[2]
            elif emission_specie in special_col2.keys():
                MEIC_tiffs_dir = 'E:\\dif_chem_con\\MEIC_EMISSION_INPUT\\RACM2\\MEIC_tiffs\\'
                print(f'{emission_specie} was change path to RACM2')
                MEIC_tiff_dir = f"{MEIC_tiffs_dir}MEIC_2020_{month}__{sector}__{species_name}.tiff"
                data_MEIC_pre = getLatLonAreaFormTiff(MEIC_tiff_dir, latmax, lonmin, latmin, lonmax)
                data_MEIC, LAT_MEIC, LON_MEIC = data_MEIC_pre[0]*special_col2[emission_specie], data_MEIC_pre[1], data_MEIC_pre[2]
            else:
                MEIC_tiffs_dir ='E:\\dif_chem_con\\MEIC_EMISSION_INPUT\\CB06\\MEIC_tiffs\\'
                MEIC_tiff_dir = f"{MEIC_tiffs_dir}MEIC_2020_{month}__{sector}__{species_name}.tiff"
                data_MEIC_pre = getLatLonAreaFormTiff(MEIC_tiff_dir, latmax, lonmin, latmin, lonmax)
                data_MEIC, LAT_MEIC, LON_MEIC = data_MEIC_pre[0], data_MEIC_pre[1], data_MEIC_pre[2]

        data_MEIC_sum = np.sum(data_MEIC)
    
        # 插值 MEIC 到目标分辨率
        for i in range(data_Temp.shape[0]):
            for j in range(data_Temp.shape[1]):
                lon = LON[i, j]
                lat = LAT[i, j]
                MEICloc_lat = getNearestPos(lat, lon, LAT_MEIC, LON_MEIC)
                if MEICloc_lat.ndim == 2: 
                    MEICloc_lat = [MEICloc_lat[0][1], MEICloc_lat[1][1]]  # 处理特殊情况采样到多个点
                data_Temp[i, j] = data_MEIC[MEICloc_lat[0], MEICloc_lat[1]]
    
        data_MEIC_interp_ = data_Temp * (data_MEIC_sum / np.sum(data_Temp)) if np.sum(data_Temp) != 0 else 0  # 总量不变
    
        # 空间分配
        predicted_grid = data_MEIC_interp_
        predicted_grid = np.array(predicted_grid)
        # if predicted_grid !=0:
        predicted_grid[np.isnan(predicted_grid)] = 0  # 设空值为0，一般在农业源的颗粒物排放会遇到
        # Convert monthly emission to weekly emission. 整月转换到周等级，以月分配系数计算，这里默认0.25 4周一月
        weekly_values = predicted_grid * 0.25
        # Convert weekly emission to daily emission. # 周分配系数转换到日，w为这周的星期几，即找到7个周分配系数中对应的某星期几
        daily_values = weekly_values * weekly_factor[int(w)]
        # 转换到小时分配，24个小时
        hourly_values = np.zeros([24, 1, ROW, COL])
        # print('sghh', hourly_values.shape)
        for hour in range(24):
            hourly_values[hour, 0, :, :] += daily_values * hourly_factor[hour]
        # 计算需要重复多少次
        repeat_times = len(periods_H) // 24  # 应该是30次
        
        # 使用 np.tile 进行重复
        hourly_values = np.tile(hourly_values, (repeat_times, 1, 1, 1))
        # hourly_values[0, 0, :, :] += daily_values * hourly_factor[int(hou)]
        # print('wdwfghgh', hourly_values.shape)
    
        # Convert original units to target units and input the split_factor. 单位转换以及split_factor加权
        origin_unit = species_info['inv_unit'][colnum]
        target_unit = species_info['emi_unit'][colnum]
        split_factor = species_info['split_factor'][colnum]
        divisor = species_info['divisor'][colnum]
        # print(origin_unit, target_unit, split_factor, divisor)
        # Convert original units to target units and input the split_factor.
        if origin_unit == "Mmol" and target_unit == "mol/s":
            hourly_values = hourly_values * 1000000.0 / 3600.0 *split_factor
        elif origin_unit == "Mg" and target_unit == "g/s":
            hourly_values = hourly_values * 1000000.0 / 3600.0 *split_factor
        elif origin_unit == "Mg" and target_unit == "mol/s":
            hourly_values = (hourly_values * 1000000.0 / 3600.0 / divisor )*split_factor
        emission_specie1 =  emission_specie  
        print(emission_specie1)
        emission_specie_var = tmpf.createVariable(emission_specie1, "f", ("TSTEP", "LAY", "ROW", "COL"))
        if target_unit == "mol/s":
            emission_specie_var.setncatts(
                dict(units="moles/s", long_name=emission_specie1, var_desc=emission_specie1))
        elif target_unit == "g/s":
            emission_specie_var.setncatts(
                dict(units="g/s", long_name=emission_specie1, var_desc=emission_specie1))
        # print('wdwfghgh', hourly_values.shape)
        emission_specie_var[:, 0, :, :] = hourly_values[:, 0, :, :]
        
    
    # Get rid of initial DUMMY variable
    del tmpf.variables["DUMMY"]

    # Update TFLAG to be consistent with variables
    tmpf.updatetflag(tstep=10000, overwrite=True)

    # Remove VAR-LIST so that it can be inferred
    delattr(tmpf, "VAR-LIST")
    tmpf.updatemeta()

    output_name = fr"{output_dir}/{target_mechanism}_{sector}_{grid_name}_{yyyymmdd}.nc"  #
    tmpf.save(output_name, format="NETCDF3_CLASSIC")
    tmpf.close()


# # if __name__ == "__main__":
# #     CMAQ_MEICAveEmission_generation(
# #             Workdir=r"E:\xyhfiles_runtest\testemission\\",  # 程序运行输出中间文件和最终结果的路径
# #             GRIDDESC_dir="E:\WCAS_serverfiles\GRIDDESC",
# #             GRIDNAME="CDsvSA_d03",
# #             GRIDCRO2D_dir="E:\WCAS_serverfiles\GRIDCRO2D_2024333.nc",
# #             sectors=['transportation', 'residential', 'power', 'agriculture', 'industry'],  # 生成的部门，MEIC5部门中选择输出哪些
# #             MEIC_dir=r"E:\SichuanCMAQPMtrends\MEIC\2020\\",  # MEIC原始ASCII文件所在路径
# #             start_date="2025-01-14",  # 生成清单的开始日期
# #             end_date="2025-01-14",  # 生成清单的结束日期
# #             factor_csvfiles_dir=r"E:\xyhfiles_runtest\factor_files\\",  # 存放分配系数的csv文件所在目录
# #     )
# #     pass

