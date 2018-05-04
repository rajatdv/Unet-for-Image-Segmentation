from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop

def conv_block(inp,filters,kernel=(3,3)):
    
    db = Conv2D(filters, kernel, padding='same')(inp)
    db = BatchNormalization()(db)
    db = Activation('relu')(db)
    db = Conv2D(filters, kernel, padding='same')(inp)
    db = BatchNormalization()(db)
    db = Activation('relu')(db)
    
    return db

def get_unet128(input_shape=(128,128,3),num_class=1):
    
    inp=Input(shape=input_shape)
    
    dn1 = conv_block(inp,64,(3,3))
    dn1_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn1)
    
    dn2 = conv_block(dn1_pool,128,(3,3))
    dn2_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn2)
    
    dn3 = conv_block(dn2_pool,256,(3,3))
    dn3_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn3)
    
    dn4 = conv_block(dn3_pool,512,(3,3))
    dn4_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn4)
    
    cn =conv_block(dn4_pool,1024,(3,3))
    
    up4 = UpSampling2D((2, 2))(cn)
    up4 = concatenate([dn4, up4], axis=3)
    up4 = conv_block(up4,512,(3,3))
    
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([dn3, up3], axis=3)
    up3 = conv_block(up3,256,(3,3))
    
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([dn2, up2], axis=3)
    up2 = conv_block(up2,128,(3,3))
    
    
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([dn1, up1], axis=3)
    up1 = conv_block(up1,64,(3,3))
    
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    
    output = Conv2D(num_class, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inp, outputs=output)
    
    return model


def get_unet256(input_shape=(256,256,3),num_class=1):
    
    inp=Input(shape=input_shape)
    
    dn0 = conv_block(inp,32,(3,3))
    dn0_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn0)
    
    dn1 = conv_block(dn0,64,(3,3))
    dn1_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn1)
    
    dn2 = conv_block(dn1_pool,128,(3,3))
    dn2_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn2)
    
    dn3 = conv_block(dn2_pool,256,(3,3))
    dn3_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn3)
    
    dn4 = conv_block(dn3_pool,512,(3,3))
    dn4_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn4)
    
    cn =conv_block(dn4_pool,1024,(3,3))
    
    up4 = UpSampling2D((2, 2))(cn)
    up4 = concatenate([dn4, up4], axis=3)
    up4 = conv_block(up4,512,(3,3))
    
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([dn3, up3], axis=3)
    up3 = conv_block(up3,256,(3,3))
    
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([dn2, up2], axis=3)
    up2 = conv_block(up2,128,(3,3))
    
    
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([dn1, up1], axis=3)
    up1 = conv_block(up1,64,(3,3))
    
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([dn0, up0], axis=3)
    up0 = conv_block(up0,32,(3,3))
    
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    
    output = Conv2D(num_class, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inp, outputs=output)
    
    return model


def get_unet512(input_shape=(512,512,3),num_class=1):
    
    inp=Input(shape=input_shape)
    
    dn0a = conv_block(inp,16,(3,3))
    dn0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn0a)
    
    dn0 = conv_block(dn0a,32,(3,3))
    dn0_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn0)
    
    dn1 = conv_block(dn0,64,(3,3))
    dn1_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn1)
    
    dn2 = conv_block(dn1_pool,128,(3,3))
    dn2_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn2)
    
    dn3 = conv_block(dn2_pool,256,(3,3))
    dn3_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn3)
    
    dn4 = conv_block(dn3_pool,512,(3,3))
    dn4_pool = MaxPooling2D((2, 2), strides=(2, 2))(dn4)
    
    cn =conv_block(dn4_pool,1024,(3,3))
    
    up4 = UpSampling2D((2, 2))(cn)
    up4 = concatenate([dn4, up4], axis=3)
    up4 = conv_block(up4,512,(3,3))
    
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([dn3, up3], axis=3)
    up3 = conv_block(up3,256,(3,3))
    
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([dn2, up2], axis=3)
    up2 = conv_block(up2,128,(3,3))
    
    
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([dn1, up1], axis=3)
    up1 = conv_block(up1,64,(3,3))
    
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([dn0, up0], axis=3)
    up0 = conv_block(up0,32,(3,3))
    
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([dn0a, up0a], axis=3)
    up0a = conv_block(up0,16,(3,3))
    
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    
    output = Conv2D(num_class, (1, 1), activation='sigmoid')(up0a)

    model = Model(inputs=inp, outputs=output)
    
    return model