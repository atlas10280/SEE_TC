from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras.optimizers import Adam

def unet_small_objects_deep(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Contracting Path
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (6,6), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (6,6), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (9,9), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (9,9), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, (12,12), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (12,12), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv4b = Conv2D(256, (15,15), activation='relu', padding='same')(pool4)
    conv4b = Conv2D(256, (15,15), activation='relu', padding='same')(conv4b)
    drop4b = Dropout(0.5)(conv4b)
    pool4b = MaxPooling2D(pool_size=(2, 2))(drop4b)

    # Bottleneck
    conv5 = Conv2D(512, (18,18), activation='relu', padding='same')(pool4b)
    conv5 = Conv2D(512, (18,18), activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Expansive Path
    up6a = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(drop5)
    merge6a = concatenate([drop4b, up6a], axis=3)
    conv6a = Conv2D(256, (15,15), activation='relu', padding='same')(merge6a)
    conv6a = Conv2D(256, (15,15), activation='relu', padding='same')(conv6a)

    up6 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6a)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(256, (12,12), activation='relu', padding='same')(merge6)
    conv6 = Conv2D(256, (12,12), activation='relu', padding='same')(conv6)
    
    up7 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128, (9,9), activation='relu', padding='same')(merge7)
    conv7 = Conv2D(128, (9,9), activation='relu', padding='same')(conv7)
    
    up8 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64, (6,6), activation='relu', padding='same')(merge8)
    conv8 = Conv2D(64, (6,6), activation='relu', padding='same')(conv8)
    
    up9 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(merge9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    
    # Output Layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model