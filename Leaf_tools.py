import ee
import geetools
import time
ee.Initialize()

def Leaf_tools(mapBounds, startDate, endDate,tilename, outputName):
    def check_ee_tasks(ee_tasks: list = []):
        for task in ee_tasks:
            taskStatus = ee.data.getTaskStatus(task.id)[0]
            print(taskStatus["description"] + ": " + taskStatus["state"])

    # Wait loop for Earth Engine tasks to complete. Polls for the task status the specificed number of seconds until it is no longer active
    def task_wait_loop(ee_task, wait_interval):
        print(ee.data.getTaskStatus(ee_task.id)[0]["description"] + ":", end=" ")
        prev_task_status = ee.data.getTaskStatus(ee_task.id)[0]["state"]
        print(prev_task_status, end=" ")
        while ee_task.active():
            task_status = ee.data.getTaskStatus(ee_task.id)[0]["state"]
            if (task_status != prev_task_status):
                print(task_status, end=" ")
            prev_task_status = task_status
            time.sleep(wait_interval)
        print(ee.data.getTaskStatus(ee_task.id)[0]["state"])    

    def s2_createFeatureCollection_estimates():
        return ee.FeatureCollection('users/rfernand387/COPERNICUS_S2_SR/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1')

    def s2_createFeatureCollection_errors():
        return ee.FeatureCollection('users/rfernand387/COPERNICUS_S2_SR/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_Error')

    def s2_createFeatureCollection_domains():
        return ee.FeatureCollection('users/rfernand387/COPERNICUS_S2_SR/weiss_or_prosail3_NNT3_Single_0_1_DOMAIN')

    def s2_createFeatureCollection_range():
        return ee.FeatureCollection('users/rfernand387/COPERNICUS_S2_SR/weiss_or_prosail3_NNT3_Single_0_1_RANGE')

    def s2_createFeatureCollection_Network_Ind():
        return ee.FeatureCollection('users/rfernand387/COPERNICUS_S2_SR/Parameter_file_sl2p')

    def s2_createImageCollection_partition():
        return ee.ImageCollection('users/rfernand387/NA_NALCMS_2015_tiles').map(
            lambda image: image.select("b1").rename("partition")).merge(
            ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V/Global").map(
                lambda image: image.select("discrete_classification").remap(
                    [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126,
                     200], [0, 8, 10, 15, 17, 16, 19, 18, 14, 13, 1, 3, 1, 5, 6, 6, 2, 4, 2, 5, 6, 6, 18],
                    0).toUint8().rename("partition")))

    def s2_createFeatureCollection_legend():
        return ee.FeatureCollection('users/rfernand387/COPERNICUS_S2_SR/Legend_sl2p')

    def l8_createFeatureCollection_estimates():
        return ee.FeatureCollection('users/rfernand387/LANDSAT_LC08_C01_T1_SR_SL2P_OUTPUT')

    def l8_createFeatureCollection_errors():
        return ee.FeatureCollection('users/rfernand387/LANDSAT_LC08_C01_T1_SR_SL2P_ERRORS')

    def l8_createFeatureCollection_domains():
        return ee.FeatureCollection('users/rfernand387/LANDSAT_LC08_C01_T1_SR/LANDSAT_LC08_C01_T1_SR_DOMAIN')

    def l8_createFeatureCollection_range():
        return ee.FeatureCollection('users/rfernand387/LANDSAT_LC08_C01_T1_SR/LANDSAT_LC08_C01_T1_SR_RANGE')

    def l8_createFeatureCollection_Network_Ind():
        return ee.FeatureCollection('users/rfernand387/LANDSAT_LC08_C01_T1_SR/Parameter_file_sl2p')

    def l8_createImageCollection_partition():
        return ee.ImageCollection('users/rfernand387/NA_NALCMS_2015_tiles').map(
            lambda image: image.select("b1").rename("partition")).merge(
            ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V/Global").map(
                lambda image: image.select("discrete_classification").remap(
                    [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126,
                     200], [0, 8, 10, 15, 17, 16, 19, 18, 14, 13, 1, 3, 1, 5, 6, 6, 2, 4, 2, 5, 6, 6, 18],
                    0).toUint8().rename("partition")))

    def l8_createFeatureCollection_legend():
        return ee.FeatureCollection('users/rfernand387/LANDSAT_LC08_C01_T1_SR/Legend_sl2p')

    # add a 'date' band: number of days since epoch
    def addDate(image):
        return image.addBands(
            ee.Image.constant(ee.Date(image.date()).millis().divide(86400000)).rename('date').toUint16())

    # computes a delta time property for an image
    def deltaTime(midDate, image):
        return ee.Image(image.set("deltaTime", ee.Number(image.date().millis()).subtract(ee.Number(midDate)).abs()))

    # mask pixels that are not clear sky in a S2 MSI image
    def s2MaskClear(image):
        qa = image.select('QA60');
        mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0));
        return image.updateMask(mask)

    # add s2 geometry bands scaled by 10000
    def addS2Geometry(colOptions, image):
        return (image.addBands(
            ee.Image.constant(0).multiply(3.1415).divide(180).cos().multiply(10000).toUint16().rename(['cosVZA']))
                .addBands(
            image.metadata(colOptions["sza"]).multiply(3.1415).divide(180).cos().multiply(10000).toUint16().rename(
                ['cosSZA']))
                .addBands(
            image.metadata(colOptions["saa"]).subtract(image.metadata(colOptions["saa"])).multiply(3.1415).divide(
                180).cos().multiply(10000).toUint16().rename(['cosRAA'])));

    # sentinel 2 land mask
    def s2MaskLand(image):
        return image.updateMask((image.select('SCL').eq(4)).Or(image.select('SCL').eq(5)))

    # returns image with selected bands scaled
    def scaleBands(bandList, scaleList, image):
        bandList = ee.List(bandList)
        scaleList = ee.List(scaleList)
        return image.addBands(srcImg=image.select(bandList).multiply(ee.Image.constant(scaleList)).rename(bandList),
                              overwrite=True)

    # Determine if inputs fall in domain of algorithm
    # Need to be updated to allow for the domain to vary with partition
    def invalidInput(sl2pDomain, bandList, image):

        sl2pDomain = ee.FeatureCollection(sl2pDomain).aggregate_array("DomainCode").sort()
        bandList = ee.List(bandList).slice(3)
        image = ee.Image(image)

        # code image bands into a single band and compare to valid codes to make QC band
        image = image.addBands(image.select(bandList)
                               .multiply(ee.Image.constant(ee.Number(10)))
                               .ceil()
                               .mod(ee.Number(10))
                               .uint8()
                               .multiply(ee.Image.constant(
            ee.List.sequence(0, bandList.length().subtract(1)).map(lambda value: ee.Number(10).pow(ee.Number(value)))))
                               .reduce("sum")
                               .remap(sl2pDomain, ee.List.repeat(0, sl2pDomain.length()), 1)
                               .rename("QC"))
        return image

    # returns image with single band named network id corresponding given
    def makeIndexLayer(image, legend, Network_Ind):

        image = ee.Image(image)  # partition image
        legend = ee.FeatureCollection(legend)  # legend to convert partition numbers to networks
        Network_Ind = ee.FeatureCollection(Network_Ind)  # legend to convert networks to networkIDs

        # get lists of valid partitions
        legend_list = legend.toList(legend.size())
        landcover = legend_list.map(lambda feature: ee.Feature(feature).getNumber('Value'))

        # get corresponding networkIDs
        networkIDs = legend_list.map(lambda feature: ee.Feature(feature).get('SL2P Network')) \
            .map(lambda propertyValue: ee.Feature(ee.FeatureCollection(Network_Ind).first()).toDictionary().getNumber(
            propertyValue))

        return image.remap(landcover, networkIDs, 0).rename('networkID')

    # Read coefficients of a network from csv EE asset
    def getCoefs(netData, ind):
        return ((ee.Feature(netData)).getNumber(ee.String('tabledata').cat(ee.Number(ind).int().format())))

    # Parse one row of CSV file for a network into a global variable
    #
    # We assume a two hidden layer network with tansig functions but
    # allow for variable nodes per layer
    def makeNets(feature, M):

        feature = ee.List(feature);
        M = ee.Number(M);

        # get the requested network and initialize the created network
        netData = ee.Feature(feature.get(M.subtract(1)));
        net = {};

        # input slope
        num = ee.Number(6);
        start = num.add(1);
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())));
        net["inpSlope"] = ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind))

        # input offset
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())))
        net["inpOffset"] = ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind))

        # hidden layer 1 weight
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())))
        net["h1wt"] = ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind))

        # hidden layer 1 bias
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())))
        net["h1bi"] = ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind))

        # hidden layer 2 weight
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())))
        net["h2wt"] = ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind))

        # hidden layer 2 bias
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())))
        net["h2bi"] = ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind))

        # output slope
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())))
        net["outSlope"] = ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind))

        # output offset
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())))
        net["outBias"] = ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind))

        return (ee.Dictionary(net))

    # Parse CSV file with list of networks for a selected variable
    # This will parse one network for each landclass partition
    def makeNetVars(asset, numNets, variableNum):

        asset = ee.FeatureCollection(asset)
        numNets = ee.Number(numNets)
        variableNum = ee.Number(variableNum)

        # get selected network
        list_features = asset.flatten()
        filtered_features = ee.FeatureCollection(asset.filter(ee.Filter.eq('tabledata3', variableNum))).toList(numNets)

        return ee.List.sequence(1, numNets).map(lambda netNum: makeNets(filtered_features, netNum))

    # returns dictionary with image masked so the networkID band equals the netIndex and the corresponding network
    def selectNet(image, netList, inputNames, netIndex):

        image = ee.Image(image)
        netList = ee.List(netList)
        inputNames = ee.List(inputNames)
        netIndex = ee.Number(netIndex).int()

        return ee.Dictionary() \
            .set("Image", ee.Image(image.updateMask(image.select('networkID').eq(netIndex)).select(inputNames))) \
            .set("Network", ee.List(netList.get(netIndex)))

        # applies two layer neural network within input and output scaling

    def applyNet(outputName, netDict):

        outputName = ee.String(outputName)
        netDict = ee.Dictionary(netDict)

        inp = ee.Image(netDict.get('Image'))
        net = ee.Dictionary(netDict.get('Network'))

        # Input scaling
        l1inp2D = inp.multiply(ee.Image(net.toArray(ee.List(['inpSlope']), 0).transpose()) \
                               .arrayProject([0]) \
                               .arrayFlatten([inp.bandNames()])) \
            .add(ee.Image(net.toArray(ee.List(['inpOffset']), 0).transpose()) \
                 .arrayProject([0]) \
                 .arrayFlatten([inp.bandNames()]))

        # Hidden layers
        l12D = ee.Image(net.toArray(ee.List(['h1wt']), 0).reshape(
            [ee.List(net.get('h1bi')).length(), ee.List(net.get('inpOffset')).length()])) \
            .matrixMultiply(l1inp2D.toArray().toArray(1)) \
            .add(ee.Image(net.toArray(ee.List(['h1bi']), 0).transpose())) \
            .arrayProject([0]).arrayFlatten([['h1w1', 'h1w2', 'h1w3', 'h1w4', 'h1w5']])

        # apply tansig 2/(1+exp(-2*n))-1
        l2inp2D = ee.Image(2).divide(ee.Image(1).add((ee.Image(-2).multiply(l12D)).exp())).subtract(ee.Image(1))

        # purlin hidden layers
        l22D = l2inp2D.multiply(ee.Image(net.toArray(ee.List(['h2wt']), 0).transpose()) \
                                .arrayProject([0]) \
                                .arrayFlatten([['h2w1', 'h2w2', 'h2w3', 'h2w4', 'h2w5']])) \
            .reduce('sum') \
            .add(ee.Image(net.toArray(ee.List(['h2bi']), 0))) \
            .arrayProject([0]) \
            .arrayFlatten([['h2bi']])

        # Output scaling
        outputBand = l22D.subtract(ee.Image(ee.Number(net.get('outBias')))) \
            .divide(ee.Image(ee.Number(net.get('outSlope'))))

        # Return network output
        return (outputBand.rename(outputName))

    # returns image with single band named networkid corresponding given
    # input partition image remapped to networkIDs
    # applies a set of shallow networks to an image based on a provided partition image band
    def wrapperNNets(network, partition, netOptions, colOptions, suffixName, imageInput):

        # typecast function parameters
        network = ee.List(network)
        partition = ee.Image(partition)
        netOptions = netOptions
        colOptions = colOptions
        suffixName = suffixName
        imageInput = ee.Image(imageInput)

        # parse partition  used to identify network to use
        partition = partition.clip(imageInput.geometry()).select(['partition'])

        # determine networks based on collection
        netList = ee.List(network.get(ee.Number(netOptions.get("variable")).subtract(1)));

        # parse land cover into network index and add to input image
        imageInput = imageInput.addBands(makeIndexLayer(partition, colOptions["legend"], colOptions["Network_Ind"]))

        # define list of input names
        return ee.ImageCollection(ee.List.sequence(0, netList.size().subtract(1)) \
                                  .map(
            lambda netIndex: selectNet(imageInput, netList, netOptions["inputBands"], netIndex)) \
                                  .map(lambda netDict: applyNet(suffixName + outputName, netDict))) \
            .max() \
            .addBands(partition) \
            .addBands(imageInput.select('networkID'))

    # returns dictionary with image masked so the networkID band equals the netIndex and the corresponding network
    def selectNet2(image, netList, inputNames, netIndex):

        image = ee.Image(image)
        netList = ee.List(netList)
        inputNames = ee.List(inputNames)
        netIndex = ee.Number(netIndex).int()
        result = ee.Dictionary() \
            .set("Image", ee.Image(image.updateMask(image.select('networkID').eq(netIndex)).select(inputNames))) \
            .set("Network", ee.List(netList.get(netIndex)))
        return result

    # applies two layer neural network within input and output scaling
    def applyNet2(outputName, netDict):

        outputName = ee.String(outputName)
        netDict = ee.Dictionary(netDict)
        inp = ee.Image(netDict.get('Image'))
        net = ee.Dictionary(netDict.get('Network'))

        # Input scaling
        l1inp2D = inp.multiply(ee.Image(net.toArray(ee.List(['inpSlope']), 0).transpose()) \
                               .arrayProject([0]) \
                               .arrayFlatten([inp.bandNames()])) \
            .add(ee.Image(net.toArray(ee.List(['inpOffset']), 0).transpose()) \
                 .arrayProject([0]) \
                 .arrayFlatten([inp.bandNames()]))

        # Hidden layers
        l12D = ee.Image(net.toArray(ee.List(['h1wt']), 0).reshape(
            [ee.List(net.get('h1bi')).length(), ee.List(net.get('inpOffset')).length()])) \
            .matrixMultiply(l1inp2D.toArray().toArray(1)) \
            .add(ee.Image(net.toArray(ee.List(['h1bi']), 0).transpose())) \
            .arrayProject([0]).arrayFlatten([['h1w1', 'h1w2', 'h1w3', 'h1w4', 'h1w5']])

        # apply tansig 2/(1+exp(-2*n))-1
        l2inp2D = ee.Image(2).divide(ee.Image(1).add((ee.Image(-2).multiply(l12D)).exp())).subtract(ee.Image(1))

        # purlin hidden layers
        l22D = l2inp2D.multiply(ee.Image(net.toArray(ee.List(['h2wt']), 0).transpose()) \
                                .arrayProject([0]) \
                                .arrayFlatten([['h2w1', 'h2w2', 'h2w3', 'h2w4', 'h2w5']])) \
            .reduce('sum') \
            .add(ee.Image(net.toArray(ee.List(['h2bi']), 0))) \
            .arrayProject([0]) \
            .arrayFlatten([['h2bi']])

        # Output scaling
        outputBand = l22D.subtract(ee.Image(ee.Number(net.get('outBias')))) \
            .divide(ee.Image(ee.Number(net.get('outSlope'))))

        # Return network output
        return (outputBand.rename(outputName))

    # Read coefficients of a network from csv EE asset
    def getCoefs2(netData, ind):
        return ((ee.Feature(netData)).getNumber(ee.String('tabledata').cat(ee.Number(ind).int().format())))

    # Parse one row of CSV file for a network into a global variable
    #
    # We assume a two hidden layer network with tansig functions but
    # allow for variable nodes per layer
    def makeNets2(feature, M):

        feature = ee.List(feature);
        M = ee.Number(M);

        # get the requested network and initialize the created network
        netData = ee.Feature(feature.get(M.subtract(1)));
        net = ee.Dictionary();

        # input slope
        num = ee.Number(6);
        start = num.add(1);
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())));
        net = net.set("inpSlope", ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind)))

        # input offset
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())));
        net = net.set("inpOffset", ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind)))

        # hidden layer 1 weight
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())));
        net = net.set("h1wt", ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind)))

        # hidden layer 1 bias
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())));
        net = net.set("h1bi", ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind)))

        # hidden layer 2 weight
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())));
        net = net.set("h2wt", ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind)))

        # hidden layer 2 bias
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())));
        net = net.set("h2bi", ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind)))

        # output slope
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())));
        net = net.set("outSlope", ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind)))

        # output offset
        num = end.add(1)
        start = num.add(1)
        end = num.add(netData.getNumber(ee.String('tabledata').cat(num.format())));
        net = net.set("outBias", ee.List.sequence(start, end).map(lambda ind: getCoefs(netData, ind)))

        return (net)

    # Parse CSV file with list of networks for a selected variable
    # This will parse one network for each landclass partition
    def makeNetVars2(asset, numNets, variableNum):

        asset = ee.FeatureCollection(asset)
        numNets = ee.Number(numNets)
        variableNum = ee.Number(variableNum)

        # get selected network
        filtered_features = ee.FeatureCollection(asset.filter(ee.Filter.eq('tabledata3', variableNum))).toList(numNets)

        # make only first net
        netList = makeNets2(filtered_features, 1)
        return netList
        # return ee.List.sequence(1,numNets).map(lambda netNum: makeNets(filtered_features,netNum))

    # returns image with single band named network id corresponding given
    def makeIndexLayer2(image, legend, Network_Ind):

        image = ee.Image(image)  # partition image
        legend = ee.FeatureCollection(legend)  # legend to convert partition numbers to networks
        Network_Ind = ee.FeatureCollection(Network_Ind)  # legend to convert networks to networkIDs

        # get lists of valid partitions
        legend_list = legend.toList(legend.size())
        landcover = legend_list.map(lambda feature: ee.Feature(feature).getNumber('Value'))

        # get corresponding networkIDs
        networkIDs = legend_list.map(lambda feature: ee.Feature(feature).get('SL2P Network')) \
            .map(lambda propertyValue: ee.Feature(ee.FeatureCollection(Network_Ind).first()).toDictionary().getNumber(
            propertyValue))

        return image.remap(landcover, networkIDs, 0).rename('networkID')

    # returns image with single band named networkid corresponding given
    # input partition image remapped to networkIDs
    # applies a set of shallow networks to an image based on a provided partition image band
    def wrapperNNets2(network, partition, netOptions, colOptions, layerName, imageInput):

        # typecast function parameters
        network = ee.List(network)
        partition = ee.Image(partition)
        netOptions = ee.Dictionary(netOptions)
        colOptions = ee.Dictionary(colOptions)
        layerName = ee.String(layerName)
        imageInput = ee.Image(imageInput)

        # parse partition  used to identify network to use
        partition = partition.clip(imageInput.geometry()).select(['partition'])

        # determine networks based on collection
        netList = ee.List(network.get(ee.Number(netOptions.get("variable")).subtract(1)))

        # parse land cover into network index and add to input image
        imageInput = imageInput.addBands(
            makeIndexLayer2(partition, colOptions.get("legend"), colOptions.get("Network_Ind")))

        # define list of input names
        netIndex = 0;
        netDict = ee.Dictionary(selectNet2(imageInput, netList, netOptions.get("inputBands"), netIndex));
        estimate = ee.Image(applyNet2(layerName, netDict))

        return estimate.addBands(partition).addBands(imageInput.select('networkID'))

    # input parameters
    # collection name
    colName = "COPERNICUS/S2_SR"

    # product name, one of('Surface_Reflectance','Albedo','FAPAR','FCOVER','LAI','CCC','CWC','DASF')
    #outputName = "LAI"

    # date range for inputs
    # startDate = ee.Date('2020-04-01')
    # endDate = ee.Date('2020-09-30')
    #year_start = year
    #year_end = year
   # month_start = 4
   # month_end = 9

    # geographical bounds of inputs you can specify a JSON geometry (e.g. from earth engine)
    #table = ee.FeatureCollection('users/ganghong/TilesSampling_Belmanip')
    # print(table)
    # Map.addLayer(table)
    #table = table.map(lambda feat: feat.buffer(-10000))
    # print(table)
    #featureNumber = ee.Number(0)
    #mapBounds = ee.FeatureCollection(ee.Feature(table.toList(table.size()).get(featureNumber))).geometry()
    # mapBounds =  ee.Geometry.Polygon( \
    #        [[[-75, 45], \
    #          [-75, 46], \
    #          [-74, 46], \
    #          [-74, 45]]])
    # other filters - for now only cloud cover
    maxCloudcover = 60

    # Output parameters , outputScale and outputOffset not applied to "Surface_Reflectance"
    outputScale = 1000
    outputOffset = 0
    exportFolder = "testSL2Pv3"
    exportScale = 20
    exportDatatype = "int"

    COLLECTION_OPTIONS = {
        'COPERNICUS/S2_SR': {
            "name": 'S2',
            "description": 'Sentinel 2A',
            "Cloudcover": 'CLOUDY_PIXEL_PERCENTAGE',
            "Watercover": 'WATER_PERCENTAGE',
            "sza": 'MEAN_SOLAR_ZENITH_ANGLE',
            "vza": 'MEAN_INCIDENCE_ZENITH_ANGLE_B8A',
            "saa": 'MEAN_SOLAR_AZIMUTH_ANGLE',
            "vaa": 'MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A',
            "VIS_OPTIONS": 'VIS_OPTIONS',
            "Collection_SL2P": ee.FeatureCollection(s2_createFeatureCollection_estimates()),
            "Collection_SL2Perrors": ee.FeatureCollection(s2_createFeatureCollection_errors()),
            "sl2pDomain": ee.FeatureCollection(s2_createFeatureCollection_domains()),
            "Network_Ind": ee.FeatureCollection(s2_createFeatureCollection_Network_Ind()),
            "partition": ee.ImageCollection(s2_createImageCollection_partition()),
            "legend": ee.FeatureCollection(s2_createFeatureCollection_legend()),
            "numVariables": 7
        },
        'LANDSAT/LC08/C01/T1_SR': {
            "name": 'L8',
            "description": 'LANDSAT 8',
            "Cloudcover": 'CLOUD_COVER_LAND',
            "Watercover": 'CLOUD_COVER',
            "sza": 'SOLAR_ZENITH_ANGLE',
            "vza": 'SOLAR_ZENITH_ANGLE',
            "saa": 'SOLAR_AZIMUTH_ANGLE',
            "vaa": 'SOLAR_AZIMUTH_ANGLE',
            "VIS_OPTIONS": 'VIS_OPTIONS',
            "Collection_SL2P": ee.FeatureCollection(l8_createFeatureCollection_estimates()),
            "Collection_SL2Perrors": ee.FeatureCollection(l8_createFeatureCollection_errors()),
            "sl2pDomain": ee.FeatureCollection(l8_createFeatureCollection_domains()),
            "Network_Ind": ee.FeatureCollection(l8_createFeatureCollection_Network_Ind()),
            "partition": ee.ImageCollection(l8_createImageCollection_partition()),
            "legend": ee.FeatureCollection(l8_createFeatureCollection_legend()),
            "numVariables": 7
        }
    }
    VIS_OPTIONS = {
        "Surface_Reflectance": {
            "COPERNICUS/S2_SR": {
                "Name": 'Surface_Reflectance',
                "description": 'Surface_Reflectance',
                "inp": ['B4', 'B5', 'B6', 'B7', 'B8A', 'B9', 'B11', 'B12']
            }
        },
        "Albedo": {
            "COPERNICUS/S2_SR": {
                "Name": 'Albedo',
                "errorName": 'errorAlbedo',
                "maskName": 'maskAlbedo',
                "description": 'Black sky albedo',
                "variable": 6,
                "inputBands": ['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
                "inputScaling": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                                 0.0001],
                "outmin": (ee.Image(ee.Array([[0]]))),
                "outmax": (ee.Image(ee.Array([[1]])))
            }
        },
        'fAPAR': {
            "COPERNICUS/S2_SR": {
                "Name": 'fAPAR',
                "errorName": 'errorfAPAR',
                "maskName": 'maskfAPAR',
                "description": 'Fraction of absorbed photosynthetically active radiation',
                "variable": 2,
                "inputBands": ['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
                "inputScaling": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                                 0.0001],
                "outmin": (ee.Image(ee.Array([[0]]))),
                "outmax": (ee.Image(ee.Array([[1]])))
            }
        },
        'fCOVER': {
            "COPERNICUS/S2_SR": {
                "Name": 'fCOVER',
                "errorName": 'errorfCOVER',
                "maskName": 'maskfCOVER',
                "description": 'Fraction of canopy cover',
                "variable": 3,
                "inputBands": ['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
                "inputScaling": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                                 0.0001],
                "outmin": (ee.Image(ee.Array([[0]]))),
                "outmax": (ee.Image(ee.Array([[1]])))
            }
        },
        'LAI': {
            "COPERNICUS/S2_SR": {
                "Name": 'LAI',
                "errorName": 'errorLAI',
                "maskName": 'maskLAI',
                "description": 'Leaf area index',
                "variable": 1,
                "inputBands": ['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
                "inputScaling": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                                 0.0001],
                "outmin": (ee.Image(ee.Array([[0]]))),
                "outmax": (ee.Image(ee.Array([[1]])))
            }
        },
        'CCC': {
            "COPERNICUS/S2_SR": {
                "Name": 'CCC',
                "errorName": 'errorCCC',
                "maskName": 'maskCCC',
                "description": 'Canopy chlorophyll content',
                "variable": 1,
                "inputBands": ['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
                "inputScaling": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                                 0.0001],
                "outmin": (ee.Image(ee.Array([[0]]))),
                "outmax": (ee.Image(ee.Array([[1000]])))
            }
        },
        'CWC': {
            "COPERNICUS/S2_SR": {
                "Name": 'CWC',
                "errorName": 'errorCWC',
                "maskName": 'maskCWC',
                "description": 'Canopy water content',
                "variable": 1,
                "inputBands": ['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
                "inputScaling": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                                 0.0001],
                "outmin": (ee.Image(ee.Array([[0]]))),
                "outmax": (ee.Image(ee.Array([[100]])))
            }
        },
        'DASF': {
            "COPERNICUS/S2_SR": {
                "Name": 'DASF',
                "errorName": 'errorDASF',
                "maskName": 'maskDASF',
                "description": 'Directional area scattering factor',
                "variable": 1,
                "inputBands": ['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
                "inputScaling": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                                 0.0001],
                "outmin": (ee.Image(ee.Array([[0]]))),
                "outmax": (ee.Image(ee.Array([[1]])))
            }
        }
    }

    # parse the networks
    colOptions = COLLECTION_OPTIONS[colName]
    netOptions = VIS_OPTIONS[outputName][colName]
    numNets = ee.Number(
        ee.Feature((COLLECTION_OPTIONS[colName]["Network_Ind"]).first()).propertyNames().remove('Feature Index').remove(
            'system:index').remove('lon').size())
    # SL2P1 = makeNetVars2(COLLECTION_OPTIONS[colName]["Collection_SL2P"],numNets,1)
    SL2P = ee.List.sequence(1, ee.Number(COLLECTION_OPTIONS[colName]["numVariables"]), 1).map(
        lambda netNum: makeNetVars(COLLECTION_OPTIONS[colName]["Collection_SL2P"], numNets, netNum))
    errorsSL2P = ee.List.sequence(1, ee.Number(COLLECTION_OPTIONS[colName]["numVariables"]), 1).map(
        lambda netNum: makeNetVars(COLLECTION_OPTIONS[colName]["Collection_SL2Perrors"], numNets, netNum))

    # make products and export

    # filter collection and add ancillary bands
    input_collection = ee.ImageCollection(colName) \
        .filterBounds(mapBounds).filterDate(startDate, endDate).filterMetadata('MGRS_TILE','equals',tilename).filterMetadata(colOptions["Cloudcover"], 'less_than',maxCloudcover) \
        .limit(5000) \
        .map(lambda image: addDate(image)) \
        .map(lambda image: image.clip(mapBounds)) \
        .map(lambda image: s2MaskClear(image)) \
        .map(lambda image: addS2Geometry(colOptions, image))
    # print(input_collection.size().getInfo())

    if outputName == "Surface_Reflectance":
        export_collection = input_collection;
    else:
        # get partition used to select network
        partition = (COLLECTION_OPTIONS[colName]["partition"]).filterBounds(mapBounds).mosaic().clip(mapBounds).rename(
            'partition')
        # pre process input imagery and flag invalid inputs
        input_collection = input_collection.map(lambda image: s2MaskLand(image)) \
            .map(lambda image: scaleBands(netOptions["inputBands"], netOptions["inputScaling"], image)) \
            .map(lambda image: invalidInput(COLLECTION_OPTIONS[colName]["sl2pDomain"], netOptions["inputBands"], image))
        ## apply networks to produce mapped parameters
        estimateSL2P = input_collection.map(
            lambda image: wrapperNNets(SL2P, partition, netOptions, COLLECTION_OPTIONS[colName], "estimate", image))
        uncertaintySL2P = input_collection.map(
            lambda image: wrapperNNets(errorsSL2P, partition, netOptions, COLLECTION_OPTIONS[colName], "error", image))
        # scale and offset mapped parameter bands
        estimateSL2P = estimateSL2P.map(lambda image: image.addBands(
            image.select("estimate" + outputName).multiply(ee.Image.constant(outputScale)).add(
                ee.Image.constant(outputOffset)), overwrite=True))
        uncertaintySL2P = uncertaintySL2P.map(lambda image: image.addBands(
            image.select("error" + outputName).multiply(ee.Image.constant(outputScale)).add(
                ee.Image.constant(outputOffset)), overwrite=True))
        # produce final export collection
        export_collection = input_collection.select(['date', 'QC']).combine(estimateSL2P).combine(uncertaintySL2P)

    return export_collection

