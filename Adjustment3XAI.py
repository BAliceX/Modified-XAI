## Guided Backpropagation(GBP), Grad-CAM(GC), Guided Grad-CAM(GGC)
import cv2
from keras import Model

class GuidedBackProp:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    


    def compute_heatmap(self, image, eps=1e-8): #epsilon(ε) male sure non-zero in denom
        guidedbackpropagationModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32) # change tensor type
            (convOutputs, predictions) = guidedbackpropagationModel(inputs)
            
            loss = predictions[:, tf.argmax(predictions[fnii])] # the probabilities of the prediction class of the specific image in all predictions image
            print('loss max, min value =', tf.reduce_max(loss),tf.reduce_min(loss))
        print('predictions[fni] = ', predictions[fnii])
        grads = tape.gradient(loss, convOutputs)
        convOutputs = convOutputs[fnii]

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
    
        guidedGrads = castConvOutputs * castGrads * grads

        guidedGrads = guidedGrads[fnii]
        print('guidedGrads AFTER select fni = ', guidedGrads.shape)

        guidedGrads = tf.reshape(guidedGrads, [1, guidedGrads.shape[0], guidedGrads.shape[1], guidedGrads.shape[2]])
        print('The guidedGrads shape AFTER reshape :', guidedGrads.shape)
        # ff = tf.Variable(tf.random_normal(f))
        output_shape = (1, 64, 64, 10)
        # output_shape_tensor = tf.constant(output_shape)


        output_feature_map = tf.nn.conv2d_transpose(guidedGrads, filters_l5, output_shape, strides=1, padding='SAME')
        # output_feature_map = tf.nn.bias_add(output_feature_map, bias_l5)
        # output_feature_map = tf.nn.relu(output_feature_map)
        output_feature_map = tf.abs(output_feature_map)
        output_feature_map = tf.squeeze(output_feature_map)
        output_feature_map = output_feature_map.numpy()

        print('convOutputs AFTER select fni shape :', convOutputs.shape)
        
        print('output_feature_map shape :', output_feature_map.shape)
        
        return output_feature_map

        
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            loss = predictions[:, tf.argmax(predictions[fnii])]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        
        convOutputs = convOutputs[fnii]
        grads = grads[fnii]
        print('grads AFTER select fni = ', grads.shape)

        weights = tf.reduce_mean(grads, axis=(0, 1))
        print('weights shape :', weights.shape)
        
        cam = tf.multiply(weights, convOutputs)
        cam = cam.numpy()
        print('cam shape :', cam.shape)

        castcam = tf.nn.relu(cam)
        print('The castcam shape BEFORE reshape :', castcam.shape)
        castcam = tf.reshape(castcam, [1, castcam.shape[0], castcam.shape[1], castcam.shape[2]])
        print('The castcam shape AFTER reshape :', castcam.shape)

        #################################### Deconvolution ####################################
        #################################### deconvolve to the fourth hidden layer (backward to fifth hidden/last layer) 5 -> 4
        output_shape_1 = (1, 64, 64, 640)
        castcamnp = castcam.numpy()

        castcamnp = tf.image.resize(castcamnp, (64, 64))
        print('castcamnp AFTER resize to 64*64', castcamnp.shape)

        castcamnp_1 = castcamnp[:,:,:,0:640]
        castcamnp_2 = castcamnp[:,:,:,640:1280]
        castcamnp_3 = castcamnp[:,:,:,1280:1920]
        castcamnp_4 = castcamnp[:,:,:,1920:2560]
        print('split castcamnp into 4 = ', castcamnp_1.shape, castcamnp_2.shape, castcamnp_3.shape, castcamnp_4.shape)
        feature_map_l1_1 = tf.nn.conv2d_transpose(castcamnp_1, filters_l1_1, output_shape_1, strides=1, padding='SAME')
        feature_map_l1_2 = tf.nn.conv2d_transpose(castcamnp_2, filters_l1_2, output_shape_1, strides=1, padding='SAME')
        feature_map_l1_3 = tf.nn.conv2d_transpose(castcamnp_3, filters_l1_3, output_shape_1, strides=1, padding='SAME')
        feature_map_l1_4 = tf.nn.conv2d_transpose(castcamnp_4, filters_l1_4, output_shape_1, strides=1, padding='SAME')
        print('feature_map_l1_1,2,3,4 =', feature_map_l1_1.shape, feature_map_l1_2.shape, feature_map_l1_3.shape, feature_map_l1_4.shape)
        output_feature_map_l1 = tf.concat((feature_map_l1_1, feature_map_l1_2, feature_map_l1_3, feature_map_l1_4),0)
        print('output_feature_map_l1 AFTER concat shape = ', output_feature_map_l1.shape)
        output_feature_map_l1 = tf.reduce_mean(output_feature_map_l1, axis=0)
        # output_feature_map_l1 = tf.nn.relu(output_feature_map_l1)
        output_feature_map_l1 = tf.abs(output_feature_map_l1)
        print('output_feature_map_l1 AFTER reduce_mean shape = ', output_feature_map_l1.shape)

        #################################### deconvolve to the third hidden layer (backward to fourth hidden layer) 4 -> 3
        output_shape_2 = (1, 64, 64, 160)
        output_feature_map_l1 = tf.reshape(output_feature_map_l1, [1, output_feature_map_l1.shape[0], output_feature_map_l1.shape[1], output_feature_map_l1.shape[2]])
        print('The output_feature_map_l1 shape AFTER reshape :', output_feature_map_l1.shape)
        output_feature_map_l1np = output_feature_map_l1.numpy()


        output_feature_map_l1np_1 = output_feature_map_l1np[:,:,:,0:160] #output_feature_map_l1np/ castcamnp
        output_feature_map_l1np_2 = output_feature_map_l1np[:,:,:,160:320]
        output_feature_map_l1np_3 = output_feature_map_l1np[:,:,:,320:480]
        output_feature_map_l1np_4 = output_feature_map_l1np[:,:,:,480:640]
        print('split into 4 = ', output_feature_map_l1np_1.shape, output_feature_map_l1np_2.shape, output_feature_map_l1np_3.shape, output_feature_map_l1np_4.shape)
        feature_map_l2_1 = tf.nn.conv2d_transpose(output_feature_map_l1np_1, filters_l2_1, output_shape_2, strides=1, padding='SAME') #2
        feature_map_l2_2 = tf.nn.conv2d_transpose(output_feature_map_l1np_2, filters_l2_2, output_shape_2, strides=1, padding='SAME')
        feature_map_l2_3 = tf.nn.conv2d_transpose(output_feature_map_l1np_3, filters_l2_3, output_shape_2, strides=1, padding='SAME')
        feature_map_l2_4 = tf.nn.conv2d_transpose(output_feature_map_l1np_4, filters_l2_4, output_shape_2, strides=1, padding='SAME')
        print('feature_map_l2_1,2,3,4 =', feature_map_l2_1.shape, feature_map_l2_2.shape, feature_map_l2_3.shape, feature_map_l2_4.shape)
        output_feature_map_l2 = tf.concat((feature_map_l2_1, feature_map_l2_2, feature_map_l2_3, feature_map_l2_4),0)
        print('output_feature_map_l2 AFTER concat shape = ', output_feature_map_l2.shape)
        output_feature_map_l2 = tf.reduce_mean(output_feature_map_l2, axis=0)
        # output_feature_map_l2 = tf.nn.relu(output_feature_map_l2)
        output_feature_map_l2 = tf.abs(output_feature_map_l2)
        print('output_feature_map_l2 AFTER reduce_mean shape = ', output_feature_map_l2.shape)

        #################################### deconvolve to the second hidden layer (backward to third hidden layer) 3 -> 2
        output_shape_3 = (1, 64, 64, 40)
        output_feature_map_l2 = tf.reshape(output_feature_map_l2, [1, output_feature_map_l2.shape[0], output_feature_map_l2.shape[1], output_feature_map_l2.shape[2]])
        print('The output_feature_map_l2 shape AFTER reshape :', output_feature_map_l2.shape)
        output_feature_map_l2np = output_feature_map_l2.numpy()


        output_feature_map_l2np_1 = output_feature_map_l2np[:,:,:,0:40] #output_feature_map_l2np/ castcamnp
        output_feature_map_l2np_2 = output_feature_map_l2np[:,:,:,40:80]
        output_feature_map_l2np_3 = output_feature_map_l2np[:,:,:,80:120]
        output_feature_map_l2np_4 = output_feature_map_l2np[:,:,:,120:160]
        print('split into 4 = ', output_feature_map_l2np_1.shape, output_feature_map_l2np_2.shape, output_feature_map_l2np_3.shape, output_feature_map_l2np_4.shape)
        feature_map_l3_1 = tf.nn.conv2d_transpose(output_feature_map_l2np_1, filters_l3_1, output_shape_3, strides=1, padding='SAME') #2
        feature_map_l3_2 = tf.nn.conv2d_transpose(output_feature_map_l2np_2, filters_l3_2, output_shape_3, strides=1, padding='SAME')
        feature_map_l3_3 = tf.nn.conv2d_transpose(output_feature_map_l2np_3, filters_l3_3, output_shape_3, strides=1, padding='SAME')
        feature_map_l3_4 = tf.nn.conv2d_transpose(output_feature_map_l2np_4, filters_l3_4, output_shape_3, strides=1, padding='SAME')
        print('feature_map_l3_1,2,3,4 =', feature_map_l3_1.shape, feature_map_l3_2.shape, feature_map_l3_3.shape, feature_map_l3_4.shape)
        output_feature_map_l3 = tf.concat((feature_map_l3_1, feature_map_l3_2, feature_map_l3_3, feature_map_l3_4),0)
        print('output_feature_map_l3 AFTER concat shape = ', output_feature_map_l3.shape)
        output_feature_map_l3 = tf.reduce_mean(output_feature_map_l3, axis=0)
        # output_feature_map_l3 = tf.nn.relu(output_feature_map_l3)
        output_feature_map_l3 = tf.abs(output_feature_map_l3)
        print('output_feature_map_l3 AFTER reduce_mean shape = ', output_feature_map_l3.shape)

        #################################### deconvolve to the first hidden layer (backward to second hidden layer) 2-> 1
        output_shape_4 = (1, 64, 64, 10)
        output_feature_map_l3 = tf.reshape(output_feature_map_l3, [1, output_feature_map_l3.shape[0], output_feature_map_l3.shape[1], output_feature_map_l3.shape[2]])
        print('The output_feature_map_l3 shape AFTER reshape :', output_feature_map_l3.shape)
        output_feature_map_l3np = output_feature_map_l3.numpy()


        output_feature_map_l3np_1 = output_feature_map_l3np[:,:,:,0:10] #output_feature_map_l3np/ castcamnp
        output_feature_map_l3np_2 = output_feature_map_l3np[:,:,:,10:20]
        output_feature_map_l3np_3 = output_feature_map_l3np[:,:,:,20:30]
        output_feature_map_l3np_4 = output_feature_map_l3np[:,:,:,30:40]
        print('split into 4 = ', output_feature_map_l3np_1.shape, output_feature_map_l3np_2.shape, output_feature_map_l3np_3.shape, output_feature_map_l3np_4.shape)
        feature_map_l4_1 = tf.nn.conv2d_transpose(output_feature_map_l3np_1, filters_l4_1, output_shape_4, strides=1, padding='SAME') #2
        feature_map_l4_2 = tf.nn.conv2d_transpose(output_feature_map_l3np_2, filters_l4_2, output_shape_4, strides=1, padding='SAME')
        feature_map_l4_3 = tf.nn.conv2d_transpose(output_feature_map_l3np_3, filters_l4_3, output_shape_4, strides=1, padding='SAME')
        feature_map_l4_4 = tf.nn.conv2d_transpose(output_feature_map_l3np_4, filters_l4_4, output_shape_4, strides=1, padding='SAME')
        print('feature_map_l4_1,2,3,4 =', feature_map_l4_1.shape, feature_map_l4_2.shape, feature_map_l4_3.shape, feature_map_l4_4.shape)
        output_feature_map_l4 = tf.concat((feature_map_l4_1, feature_map_l4_2, feature_map_l4_3, feature_map_l4_4),0)
        print('output_feature_map_l4 AFTER concat shape = ', output_feature_map_l4.shape)
        output_feature_map_l4 = tf.reduce_mean(output_feature_map_l4, axis=0)
        # output_feature_map_l4 = tf.nn.relu(output_feature_map_l4)
        output_feature_map_l4 = tf.abs(output_feature_map_l4)
        print('output_feature_map_l4 AFTER reduce_mean shape = ', output_feature_map_l4.shape)

        #################################### deconvolve to the INPUT layer (backward to first hidden layer) 1 -> Input
        output_shape_5 = (1, 64, 64, 10)
        output_feature_map_l4 = tf.reshape(output_feature_map_l4, [1, output_feature_map_l4.shape[0], output_feature_map_l4.shape[1], output_feature_map_l4.shape[2]])
        print('The output_feature_map_l4 shape AFTER reshape :', output_feature_map_l4.shape)
        output_feature_map_l4np = output_feature_map_l4.numpy()
        print('output_feature_map_l4np = ', output_feature_map_l4np.shape)
        feature_map_l5_1 = tf.nn.conv2d_transpose(output_feature_map_l4np, filters_l5, output_shape_5, strides=1, padding='SAME')

        # feature_map_l5_1 = tf.nn.conv2d_transpose(castcamnp, filters_l5_1, output_shape_5, strides=1, padding='SAME') #backward to first hidden layer
        # feature_map_l5_1 = tf.nn.bias_add(feature_map_l5_1, bias_l5) # adding bias
        # feature_map_l5_1 = tf.nn.relu(feature_map_l5_1) # do relu (catch >0 values)
        feature_map_l5_1 = tf.abs(feature_map_l5_1)
        print('feature_map_l5_1 =', feature_map_l5_1.shape)
        
        finalcamnp = tf.squeeze(feature_map_l5_1)
        finalcamnp = finalcamnp.numpy() #castcam.numpy()
                
        return finalcamnp #castcamnp #cam_heatmap