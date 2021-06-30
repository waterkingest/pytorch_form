import os,sys
from osgeo import gdal, gdalconst
from osgeo import ogr
from osgeo import osr
import numpy
#读取shap文件
def readShap(filename):
    #为了支持中文路径，请添加下面这句代码 
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")  
    #为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCODING","")  
    #注册所有的驱动 
    ogr.RegisterAll()  
    #数据格式的驱动
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(filename,1);
    if ds is None:
        print ('Could not open '+filename)
        sys.exit(1)
    #获取第0个图层
    layer0 = ds.GetLayerByIndex(0);
    #投影
    spatialRef = layer0.GetSpatialRef();
    # 输出图层中的要素个数  
    print('要素个数=', layer0.GetFeatureCount(0))  
    print('属性表结构信息')  
    defn = layer0.GetLayerDefn()  
    iFieldCount = defn.GetFieldCount()  
    # for index in range(iFieldCount):  
    #     oField =defn.GetFieldDefn(index)  
    #     print( '%s: %s(%d.%d)' % (oField.GetNameRef(),oField.GetFieldTypeName(oField.GetType()),oField.GetWidth(),oField.GetPrecision()))  
    indexB = defn.GetFieldIndex('ID')
    # 下面开始遍历图层中的要素  
    i=0
    for feature in layer0:  
        
        i+=1
        # 获取要素中的属性表内容  
        if feature.GetFieldAsInteger(indexB)==30:
            feature.SetField2('ID', 255)
        if feature.GetFieldAsInteger(indexB)==255:
            print ('chuange success')
        layer0.SetFeature(feature)       
        feature = None
        # 获取要素中的几何体  
        # geometry =feature.GetGeometryRef()  
        # print (geometry)
        # 为了演示，只输出一个要素信息  
        #break  
    ds.Destroy()
def shape_to_raster(shapefile, rasterfile, savefile):
    data = gdal.Open(rasterfile, gdal.GA_ReadOnly)
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    shape = ogr.Open(shapefile)
    layer = shape.GetLayer()
    targetDataset = gdal.GetDriverByName('GTiff').Create(savefile, x_res, y_res, 1, gdal.GDT_Byte)
    targetDataset.SetGeoTransform(data.GetGeoTransform())
    targetDataset.SetProjection(data.GetProjection())
    band = targetDataset.GetRasterBand(1)
    NoData_value = -9999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(targetDataset, [1], layer,options=["ATTRIBUTE=ID"] )
    # gdal.RasterizeLayer(targetDataset, [1], layer,burn_values=[0] )
def main():
    filepath=r'F:\exciseData\water-20\shp'#shp形式label位置
    rasterfilepath=r'F:\exciseData\water-20\img'
    savepath=r'F:\exciseData\water-20\changelabel'
    filelist=os.listdir(filepath)
    for i in filelist:
        if i.endswith('.shp'):
            shapefile=filepath+'\\'+i
            rasterfile=rasterfilepath+'\\'+i.replace('.shp','.tif')
            savefile=savepath+'\\'+i.replace('.shp','.png')
            shape_to_raster(shapefile, rasterfile, savefile)
            # readShap(shapefile)
            # break
    # readShap();
    
if __name__ == "__main__":
    main();