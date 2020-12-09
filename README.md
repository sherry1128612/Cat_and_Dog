@[Cat-and-Dog]
# Cat-and-Dog
The exercise of deep learning

## 环境配置
    华为云
    Python3
    GPU:2*v100NV32 CPU: 16 核 128GiB
    Tensorflow 1.8
    keras 2.2.4

## 深度学习具体算法
### 构建卷积网络，从头开始训练
#### 查看notebook工作路径
    cd .
    !ls
#### 将OBS中存储的样本数据下载至noteboook
    '''
        bucket_path：OBS中数据集存储具体路径
    '''
    from modelarts.session import Session
    session = Session()
    session.download_data(bucket_path="/cat-dog.dazhan/cat-dog/data/train/", path="/home/ma-user/work/")  
    !ls

#### 引入库源
    import tensorflow as tf
    import numpy as np
    import os
    import matplotlib.pyplot as plt

### 用0，1分别标记猫狗图片
    def get_files(file_dir):
        #file_dir: 文件夹路径
        #return: 乱序后的图片和标签
    
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    #载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] =='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_cats.append(1)
    print("There are %d cats\nThere are %d dogs" % (len(cats),len(dogs)))
    
    #打乱文件顺序
    image_list = np.hstack((cats,dogs))
    label_list = np.hstack((label_cats,label_dogs))
    temp = np.array([image_list,label_list])
    temp = temp.transpose()    #转置
    np.random.shuffle(temp)    #利用shuffle打乱顺序
    
    #从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]   #字符串类型转换为int类型
    
    return image_list,label_list

    #生成相同大小的批次
    def get_batch(image,label,image_W,image_H,batch_size,capacity):
        #image,label:要生成batch的图像和标签list
        #image_W,image_H:图片的宽高
        #batch_size:每个batch有多少张图片
        #capacity:队列容量，一个队列最大多少
        #return:图像和标签的batch
    
    #将python.list类型转换成tf能够识别的格式
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int64)
    
    #生成队列，将image和label放到队列里
    input_queue = tf.train.slice_input_producer([image,label])
    
    image_contents = tf.read_file(input_queue[0])   #读取图片的全部信息
    label = input_queue[1]
    #将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等
    #把图片解码，channels = 3为彩色图片，r，g，b；黑白图片为1，也可以理解为图片的厚度
    image = tf.image.decode_jpeg(image_contents,channels = 3)
    
    #统一图片大小
    #将图片以图片中心进行裁剪或者扩充为指定的image_W,image_H
    image = tf.image.resize_images(image,[image_H,image_W],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)   #最近邻插值方法
    image = tf.cast(image,tf.float32)      #string类型转换为float    
    image = image/255
    #image = tf.image.per_image_standardization(image)     #对数据进行标准化，就是减去它的均值，除以其方差
    #生成批次num_threads有多少个线程根据电脑配置设置  capacity队列中，最多容纳图片的个数
    #tf.train.shuffle_batch  打乱顺序
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size = batch_size,
                                             num_threads = 64,
                                             capacity = capacity)
    
    return image_batch,label_batch

#### 将猫、狗图片分开存储至三个文件夹   
    import os, shutil
    #The path to the directory where the original
    #dataset was uncompressed
    #original_dataset_dir = './test/',根据数据集所存文件夹进行修改
    original_dataset_dir = './train/dog_and_cat_200/'
    #The directory where we will
    #store our smaller dataset
    #base_dir = './test/base/'
    base_dir = './train/base/'
    os.mkdir(base_dir)
    #Directories for our training,
    #validation and test splits
    train_dir1 = os.path.join(base_dir, 'train')
    os.mkdir(train_dir1)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)
    #Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir1, 'cats')
    os.mkdir(train_cats_dir)
    #Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir1, 'dogs')
    os.mkdir(train_dogs_dir)
    #Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    #Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)
    #Directory with our validation cat pictures
    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)
    #Directory with our validation dog pictures
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)
    #Copy first 60 cat images to train_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(60)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    #Copy next 20 cat images to validation_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(60, 80)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    #Copy next 20 cat images to test_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(80, 100)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)    
    #Copy first 60 dog images to train_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(60)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)    
    #Copy next 20 dog images to validation_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(60, 80)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)    
    #Copy next 20 dog images to test_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(80, 100)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
  
#### 数据增强
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    #This is module with image preprocessing utilities
    from keras.preprocessing import image
    fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
    #We pick one image to "augment"
    img_path = fnames[3]
    #Read the image and resize it
    img = image.load_img(img_path, target_size=(150, 150))
    #Convert it to a Numpy array with shape (150, 150, 3)
    x = image.img_to_array(img)
    #Reshape it to (1, 150, 150, 3)
    x = x.reshape((1,) + x.shape)
    #The .flow() command below generates batches of randomly transformed images.
    #It will loop indefinitely, so we need to `break` the loop at some point!
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break
    plt.show()

#### 创建卷积网络
    from keras import layers
    from keras import models
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    from keras import optimizers
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc'])

#### 调用网络，开始训练与测试
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)
    #Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            train_dir1,
            # All images will be resized to 150x150
            target_size=(150, 150),
            batch_size=32,
            #Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=100)
      
#### 绘制损失与准确度的结果图  
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

### 预训练模型VGG16
#### 加载VGG16预训练模型
    from keras.applications import VGG16
    conv_base = VGG16(weights = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top = False, input_shape=(150, 150, 3))
    conv_base.summary()
    #将图片导入
    import os 
    import numpy as np
    from keras.preprocessing.image import ImageDataGenerator
    base_dir = './train/base/'      #依据具体路径进行调整
    train_dir1 = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    datagen = ImageDataGenerator(rescale = 1. / 255)
    batch_size = 20
    def extract_features(directory, sample_count):
        features = np.zeros(shape = (sample_count, 4, 4, 512))
        labels = np.zeros(shape = (sample_count))
        generator = datagen.flow_from_directory(directory, target_size = (150, 150), 
                                                batch_size = batch_size,
                                                class_mode = 'binary')
        i = 0
        for inputs_batch, labels_batch in generator:
            #把图片输入VGG16卷积层，让它把图片信息抽取出来
            features_batch = conv_base.predict(inputs_batch)
            #feature_batch 是 4*4*512结构
            features[i * batch_size : (i + 1)*batch_size] = features_batch
            labels[i * batch_size : (i+1)*batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count :
                #for in 在generator上的循环是无止境的，因此我们必须主动break掉
                break
            return features , labels
    #extract_features 返回数据格式为(samples, 4, 4, 512)
    train_features, train_labels = extract_features(train_dir1, 120)
    validation_features, validation_labels = extract_features(validation_dir, 40)
    test_features, test_labels = extract_features(test_dir, 40)

#### 构造网络开始训练
    train_features = np.reshape(train_features, (120, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (40, 4 * 4 * 512))
    test_features = np.reshape(test_features, (40, 4 * 4* 512))
    from keras import models
    from keras import layers
    from keras import optimizers
    #构造我们自己的网络层对输出数据进行分类
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim = 4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr = 2e-5), loss = 'binary_crossentropy', metrics = ['acc'])
    history = model.fit(train_features, train_labels, epochs = 30, batch_size = 20, 
                        validation_data = (validation_features, validation_labels))

#### 绘制损失与准确度的结果图
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

### VGG16的卷积层添加至自主搭建的网络
#### 优化结构
    from keras import layers
    from keras import models
    model = models.Sequential()
    #将VGG16的卷积层直接添加到我们的网络
    model.add(conv_base)
    #添加我们自己的网络层
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.summary()
    #参数调优，冻结层
    conv_base.trainable = True
    set_trainable = False
    #一旦读取到'block5_conv1'时，意味着来到卷积网络的最高三层
    #可以使用conv_base.summary()来查看卷积层的信息
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            #当trainable == True 意味着该网络层可以更改，要不然该网络层会被冻结，不能修改
            layer.trainable = True
        else:
            layer.trainable = False
    #把图片数据读取进来
    test_datagen = ImageDataGenerator(rescale = 1. / 255)
    train_generator = test_datagen.flow_from_directory(train_dir1, target_size = (150, 150), batch_size = 20,
                                                    class_mode = 'binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size = (150,150),
                                                        batch_size = 20,
                                                        class_mode = 'binary')
    model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(2e-5),
                metrics = ['acc'])

    history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 30, 
                                validation_data = validation_generator,
                                validation_steps = 50)
#### 绘制损失与准确度的结果图
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()                              
                              
                           
    
