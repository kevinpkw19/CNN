from timeit import Timer
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import time
from tensorflow.keras.utils import to_categorical
import numpy.lib.stride_tricks as nx
from timeit import default_timer as timer


#loads data and performs some pre-processing
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
Y_train = to_categorical(Y_train) #converts it to one-hot encoding starting with 0 and ending with 9
Y_test = to_categorical(Y_test)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train , X_test = X_train/255.0 , X_test/255.0

#defines the NN's parameters
class NeuralNet():
    def __init__(self,inputs,input_truth,filter_1_dim=(16,3,3),filter_2_dim=(32,16,3,3),max_pool_layers=2,categories=10):
        self.filter_1_dim=filter_1_dim
        self.filter_2_dim=filter_2_dim
        self.input=inputs
        self.input_truth=input_truth
        i_s,i_r,i_c= inputs.shape
        final_connections= i_r//(2**max_pool_layers) * i_c//(2**max_pool_layers) * filter_2_dim[0]
        s,r,c=filter_1_dim
        self.w_1=np.random.randn(final_connections,final_connections//2) * np.sqrt(2/final_connections)
        self.w_2=np.random.randn(final_connections//2,final_connections//4)* np.sqrt(2/(final_connections//2))
        self.w_3=np.random.randn(final_connections//4,categories)* np.sqrt(2/(final_connections//4))
        self.round_16=np.random.randn(s,r,c)
        ss,s,r,c=filter_2_dim
        self.round_32=np.random.randn(ss,s,r,c)


    #makes minibatches based on the number of mini-batches desired
    def make_mini_batch(self,num_mini_batches):
        size=self.input.shape[0]//num_mini_batches
        all_mini_batches=[]
        all_truth_mini_batches=[]
        for i in range(num_mini_batches):
            new_mini_batch=self.input[i*size:(i*size)+size,:,:]
            new_answers_mini_batch=self.input_truth[i*size:(i*size)+size]
            all_mini_batches.append(new_mini_batch)
            new_answers_mini_batch=np.array(new_answers_mini_batch)
            all_truth_mini_batches.append(new_answers_mini_batch)
        all_truth_mini_batches=np.array(all_truth_mini_batches)
        return all_mini_batches,all_truth_mini_batches

    #Leaky ReLU function
    def Relu(self,input):
        return max(0.01*input,input)
    #softmax activation
    def softmax(self):
        e_x = np.exp(self.output_layer - np.max(self.output_layer))
        return e_x / e_x.sum()


    #convolution layer 1 which uses a sliding window
    def convolution_1(self,input,filter,stride=1):
        
        i_s,i_r,i_c=input.shape
        f_s,f_r,f_c=filter.shape

        input=input.reshape(i_r,i_c)
        filter=filter.reshape(f_r,f_c)
        input= np.pad(input, 1, mode='constant')
        feature_map=[]
        a=nx.sliding_window_view(input, (f_r,f_c))

        a=np.squeeze(a)
        
        a=a.reshape(a.shape[0]*a.shape[1],a.shape[2],a.shape[3])
        feature_map=np.tensordot(a,filter,axes=2)
        feature_map=feature_map.reshape(i_r,i_c)

        return feature_map


    #convolution layer 2 which uses a sliding window
    def convolution_2(self,input,filter,stride=1):
        i_s,i_r,i_c=input.shape
        f_ss,f_s,f_r,f_c=filter.shape

        input=input.reshape(i_s,i_r,i_c)
        filter=filter.reshape(f_s,f_r,f_c)

        p_width=((0,0),(1,1),(1,1))
        input=np.pad(input,pad_width=p_width,mode='constant',constant_values=0)


        feature_map=[]

        a=nx.sliding_window_view(input, (f_s,f_r,f_c))
        a=np.squeeze(a)
        
        a=a.reshape(a.shape[0]*a.shape[1],a.shape[2],a.shape[3],a.shape[4])
        feature_map=np.tensordot(a,filter,axes=3)
        feature_map=feature_map.reshape(i_r,i_c)

        return feature_map
 
    #  full convolution layer which includes the cycling of each filter over the image
    def full_convolution_1(self,image_data,stride=1):
        self.stacked_feature_maps_16=[]
        total_filters=self.round_16.shape[0]

        for i in range(total_filters):# level 1 filters
            filter=self.round_16[i:i+1,:,:]
            featuremap=self.convolution_1(image_data,filter)
            self.stacked_feature_maps_16.append(featuremap)
        
        self.stacked_feature_maps_16=np.array(self.stacked_feature_maps_16)
        return self.stacked_feature_maps_16

    def full_convolution_2(self,image_data,stride=1):
        self.stacked_feature_maps_32=[]
        total_filters=self.round_32.shape[0]


        for i in range(total_filters):# level 2 filters
            filter=self.round_32[i:i+1,:,:,:]
            featuremap=self.convolution_2(image_data,filter)
            self.stacked_feature_maps_32.append(featuremap)

        
        self.stacked_feature_maps_32=np.array(self.stacked_feature_maps_32)

        return self.stacked_feature_maps_32

    #max-pools the output at each of the convolution layers
    #also returns a list of tuples, each of which indicate which pixel was chosen as the max from each 2x2 grouping.
    def pooling(self,stacked_feature_map,pooled_dim=0):
        sf_s,sf_r,sf_c= stacked_feature_map.shape
        stacked_pooled_maps=[]
        stacked_value_index_list=[]
        
        for i in range(sf_s):
            feature_map=stacked_feature_map[i:i+1,:,:]
            pooled_map=[]
            single_value_index_list=[]
            count_r=0
           
            for r in range(sf_r//2 ):
                row=[]
                count_c=0
                for c in range(sf_c//2 ):
                    current_map=feature_map[0,count_r:count_r+2,count_c:count_c+2]
                    index= np.argmax(current_map)
                    unmod_index= np.unravel_index(index,current_map.shape)
                    value=current_map[unmod_index[0]][unmod_index[1]]
                    proper_index=(unmod_index[0]+count_r,unmod_index[1]+count_c)#useful for backprop
                    single_value_index_list.append(proper_index)
                    row.append(value)
                    count_c+=2
                count_r+=2
                pooled_map.append(row)
            stacked_pooled_maps.append(pooled_map)

            stacked_value_index_list.append(single_value_index_list)

        stacked_pooled_maps=np.array(stacked_pooled_maps).reshape(sf_s,sf_r//2,sf_c//2)

        stacked_value_index_list=np.array(stacked_value_index_list,dtype="i") 

        return stacked_pooled_maps,stacked_value_index_list
        


    


    #backprops the error from the softmax layer back to the 3rd set of weights
    def backprop_for_weights_3(self,y):
        predicted=self.softmax()
        new_gradients=[]
        de_wr_dz=(predicted-y).flatten() #gives error with respect to input of softmax function with one hot encoding and bce. need to find partial deriv of input wr to weight
        for i in range(len(de_wr_dz)):
            error_to_input=de_wr_dz[i]
            w=self.fc_2*error_to_input
            new_gradients.append(w)
        new_gradients=np.array(new_gradients)
        new_gradients=np.transpose(new_gradients)
        return new_gradients

    #backprops the error from the 2nd hidden layer back to the 2nd set of weights
    def backprop_for_weights_2(self,prev_error_node,fc_layer):
        new_gradients=[]
        for i in range(prev_error_node.shape[0]):
            error_to_input=prev_error_node[i]
            w=error_to_input*fc_layer
            new_gradients.append(w)
        new_gradients=np.array(new_gradients)
        new_gradients=np.transpose(new_gradients)

        return new_gradients


    #backprops the error from the 1st hidden layer back to the 1st set of weights
    def backprop_for_weights_1(self,prev_error_node,flattened_sfm):
        new_gradients=[]
        for i in range(prev_error_node.shape[0]):
            error_to_input=prev_error_node[i]
            w=error_to_input*flattened_sfm
            new_gradients.append(w)
        new_gradients=np.array(new_gradients)
        new_gradients=np.transpose(new_gradients)

        return new_gradients

    #backprops the error from the softmax layer back to the 2nd FC layer
    def backprop_for_fully_con_2(self,y):
        
        predicted=self.softmax()
        avg_error=[]
        de_wr_dz=(predicted-y).flatten()
        for u in range(len(de_wr_dz)):
            error_to_input=de_wr_dz[u]
            weight_used=self.w_3[:,u:u+1]
            layer=error_to_input*weight_used
            avg_error.append(layer)
        avg_error=np.array(avg_error)
        avg_error=np.squeeze(avg_error)
        avg_error=avg_error.transpose()
        error_per_node_for_fully_con=avg_error.mean(axis=1)

                
        return error_per_node_for_fully_con
        
    #backprops the error from the 2nd FC layer back to the 1st FC layer
    def backprop_for_fully_con_1(self,prev_error_node):
        avg_error=[]
        for i in range(prev_error_node.shape[0]):
            error_to_input=prev_error_node[i]
            weight_used=self.w_2[:,i:i+1]
            layer=error_to_input*weight_used
            avg_error.append(layer)
        avg_error=np.array(avg_error)
        avg_error=np.squeeze(avg_error)
        avg_error=avg_error.transpose()
        error_per_node_for_fully_con=avg_error.mean(axis=1)


        return error_per_node_for_fully_con

    
    #backprops the error from the 1st FC layer back to the flattened SFM layer
    def backprop_for_flattened_sfm(self,prev_error_node):
        avg_error=[]
        for i in range(prev_error_node.shape[0]):
            error_to_input=prev_error_node[i]
            weight_used=self.w_1[:,i:i+1]
            layer=error_to_input*weight_used
            avg_error.append(layer)
        avg_error=np.array(avg_error)
        avg_error=np.squeeze(avg_error)
        avg_error=avg_error.transpose()
        error_per_node_for_flattened_sfm=avg_error.mean(axis=1)


        return error_per_node_for_flattened_sfm


    def backprop_maxpool_2(self,stacked_feature_map_dim,error_per_node_for_flattened_sfm,stacked_value_index_list):
         # finds error for each node in the pre-pooled layer but after relu. first looks into cache of indices extracted during 
         # max pool and then uses it to figure out which index has errors and which is 0
         #Does this by going into each 2x2 portion of pre-pooled feature map then flattened and comparing it to indices used for max pool.
            
        f_s,f_r,f_c= stacked_feature_map_dim
        stacked_error_feature_maps=[]
        error_per_node_for_flattened_sfm=error_per_node_for_flattened_sfm.reshape(f_s*f_r//2*f_c//2)
        counter=0
        for i in range(f_s):
            error_feature_map=np.zeros((f_r, f_c))
            ref_list=stacked_value_index_list[i:i+1,:]
 
            ref_list=ref_list.reshape(((f_c//2)**2),2)
            for location in range((f_c//2)**2):
                ref=ref_list[location]
                error_feature_map[ref[0],ref[1]]=error_per_node_for_flattened_sfm[counter]
                counter+=1
            stacked_error_feature_maps.append(error_feature_map)

        stacked_error_feature_maps=np.array(stacked_error_feature_maps).reshape(f_s,f_r,f_c)
        return stacked_error_feature_maps


    def backprop_maxpool_1(self,stacked_feature_map_dim,error_per_node_for_conv2_layer,stacked_value_index_list):
         # finds error for each node in the pre-pooled layer but after relu. first looks into cache of indices extracted during 
         # max pool and then uses it to figure out which index has errors and which is 0
         #Does this by going into each 2x2 portion of pre-pooled feature map then flattened and comparing it to indices used for max pool.

        f_s,f_r,f_c= stacked_feature_map_dim
        stacked_error_feature_maps=[]
        error_per_node_for_conv2_layer=error_per_node_for_conv2_layer.reshape(f_s*f_r//2*f_c//2)
        counter=0
        for i in range(f_s):
            error_feature_map=np.zeros((f_r, f_c))
            ref_list=stacked_value_index_list[i:i+1,:]
            ref_list=ref_list.reshape(((f_c//2)**2),2)
            for location in range((f_c//2)**2):
                ref=ref_list[location]
                error_feature_map[ref[0]][ref[1]]=error_per_node_for_conv2_layer[counter]
                counter+=1
            stacked_error_feature_maps.append(error_feature_map)

        stacked_error_feature_maps=np.array(stacked_error_feature_maps).reshape(f_s,f_r,f_c)
        return stacked_error_feature_maps


    #backprops the errors back to the 2nd layer of 3x3 convolutional filters
    def backprop_filter_2(self,stacked_error_feature_maps,input):
        input=input.reshape(16,14,14)
        filter_ss,filter_s,filter_r,filter_c=self.filter_2_dim
        total_filter_errors=[]
        for ss in range(filter_ss):
            single_filter_errors=[]
            for d in range(filter_s):
                layer_weights=[]
                for r in range(filter_r):
                    row_error=[]
                    for c in range(filter_c):
                        error=0
                        output_wr_to_w= input[d:d+1,r:r+filter_r,c:c+filter_c].flatten()
                        output=stacked_error_feature_maps[ss:ss+1,r:r+filter_r,c:c+filter_c].flatten()
                        for i in range(output_wr_to_w.shape[0]):
                            if output[i]!=0:
                                error=error+(output[i]*output_wr_to_w[i])
        
                        row_error.append(error)
                
                    layer_weights.append(row_error)
                layer_weights=np.array(layer_weights).reshape(3,3)
                single_filter_errors.append(layer_weights)
            single_filter_errors=np.array(single_filter_errors)
            total_filter_errors.append(single_filter_errors)
        total_filter_errors=np.array(total_filter_errors)
    
        return total_filter_errors

    #backprops the errors back to the 1st layer of 3x3 convolutional filters
    def backprop_filter_1(self,stacked_error_feature_maps,input):

        input=input.reshape(28,28)
        filter_s,filter_r,filter_c=self.filter_1_dim

        filter_errors=[]
        for d in range(filter_s):
            d_2_weights=[]
            for r in range(filter_r):
                row_error=[]
                for c in range(filter_c):
                    error=0
 
                    output_wr_to_w= input[r:r+filter_r,c:c+filter_c].flatten()
                 
                    output=stacked_error_feature_maps[d:d+1,r:r+filter_r,c:c+filter_c].flatten()
      
                    for i in range(output_wr_to_w.shape[0]):
                        if output[i]!=0:
                            error=error+(output[i]*output_wr_to_w[i])
       
                    row_error.append(error)
                
                d_2_weights.append(row_error)
            d_2_weights=np.array(d_2_weights).reshape(3,3)
            filter_errors.append(d_2_weights)
        filter_errors=np.array(filter_errors)
    
        return filter_errors



    def error_for_input_conv2(self,stacked_error_feature_maps):
        mod_stacked_error_feature_maps=np.c_[stacked_error_feature_maps,np.zeros((32,14,2))]
        temp_kernel=np.copy(self.round_32)

        f_ss,f_s,f_r,f_c=temp_kernel.shape #filter (32,16,3,17)
        #run through all 16 filter layers over 1 single layer of mod_stacked. then repeat 32 times and average it out
        #flattens the input and kernel. and inserts 0's into the kernel to create a pseudo-sliding view over the input for backpropping error
        final_list=[]
        for i in range(f_ss):
            input=mod_stacked_error_feature_maps[i:i+1,:,:]
            input=input.flatten()
            input=np.delete(input,[input.shape[0]-1,input.shape[0]-2])
            temp_list=[]
            for u in range(f_s):
                kernel=temp_kernel[i:i+1,u:u+1,:,:]
                kernel=kernel.flatten()
                kernel=np.insert(kernel,3,np.zeros(13))
                kernel=np.insert(kernel,19,np.zeros(13))
                test=np.convolve(input,kernel,'full')

                test=test.reshape(16,16)
                temp_list.append(test)
            temp_list=np.array(temp_list)
            final_list.append(temp_list)
        final_list=np.array(final_list)

        final_list=final_list[:,:,1:-1,1:-1]

        final_list=np.mean(final_list,axis=0)


        return final_list
      
    #function to run through the entire training process
    def fcl_1(self,mini_batch,truth_mini_batch,learning_rate=0.005):
        new_weight_grad_w3=np.zeros(self.w_3.shape)
        new_weight_grad_w2=np.zeros(self.w_2.shape)
        new_weight_grad_w1=np.zeros(self.w_1.shape)
        new_filter_grad=np.zeros(self.filter_1_dim)
        new_filter_grad_2=np.zeros(self.filter_2_dim)
        d1 = np.random.binomial(1, 0.7, size=self.w_1.shape[1])
        d2 = np.random.binomial(1, 0.7, size=self.w_2.shape[1])
        correct=0
        loss=0
        for i in range(mini_batch.shape[0]):
         
            cur_sample=mini_batch[i:i+1,:,:]
            cur_truth= truth_mini_batch[i:i+1]
            cur_truth=cur_truth.reshape(10)

            sfm_1=self.full_convolution_1(cur_sample)
            pooled_sfm_1,stacked_value_index_list_1=self.pooling(sfm_1)
            sfm_2=self.full_convolution_2(pooled_sfm_1)
            pooled_sfm_2,stacked_value_index_list_2=self.pooling(sfm_2)

            
            sfm_s,sfm_r,sfm_c=pooled_sfm_2.shape
            flattened_sfm= pooled_sfm_2.reshape(sfm_r*sfm_c*sfm_s)
            dot_prod=np.dot(self.w_1.transpose(),flattened_sfm)
            self.fc_1=np.maximum(dot_prod,dot_prod*0.01)
            self.fc_1=(np.multiply(self.fc_1,d1))*0.7
            dot_prod_2=np.dot(self.w_2.transpose(),self.fc_1)
            self.fc_2=np.maximum(dot_prod_2,dot_prod_2*0.01)
            self.fc_2=(np.multiply(self.fc_2,d2))*0.7
            dot_prod_3=np.dot(self.w_3.transpose(),self.fc_2)
            self.output_layer=np.maximum(dot_prod_3,dot_prod_3*0.01)
            predicted_list=self.softmax()
            predicted=np.argmax(predicted_list)
            predicted_y=np.argmax(cur_truth)
            
     
            loss=loss+(predicted_list[predicted_y]-cur_truth[predicted_y])
           

            if predicted==predicted_y:
                correct+=1
           
            
            new_weight_grad_w3=new_weight_grad_w3+self.backprop_for_weights_3(cur_truth)
            error_for_full_con_2=self.backprop_for_fully_con_2(cur_truth)

            new_weight_grad_w2=new_weight_grad_w2+self.backprop_for_weights_2(error_for_full_con_2,self.fc_1)
            error_for_full_con_1=self.backprop_for_fully_con_1(error_for_full_con_2)
            new_weight_grad_w1=new_weight_grad_w1+self.backprop_for_weights_1(error_for_full_con_1,flattened_sfm)
            error_per_node_for_flattened_sfm=self.backprop_for_flattened_sfm(error_for_full_con_1)
            stacked_error_feature_maps=self.backprop_maxpool_2(sfm_2.shape,error_per_node_for_flattened_sfm,stacked_value_index_list_2)

            
            new_filter_grad_2=new_filter_grad_2+self.backprop_filter_2(stacked_error_feature_maps,pooled_sfm_1)

            backpropped_error=self.error_for_input_conv2(stacked_error_feature_maps)


            stacked_error_feature_maps=self.backprop_maxpool_1(sfm_1.shape,backpropped_error,stacked_value_index_list_1)
            new_filter_grad=new_filter_grad+self.backprop_filter_1(stacked_error_feature_maps,cur_sample)
 
           
        loss=loss/mini_batch.shape[0]
        avg_filter_grad=new_filter_grad/mini_batch.shape[0]
        avg_filter_grad_2=new_filter_grad_2/mini_batch.shape[0]
        avg_Weight_grad_w1=new_weight_grad_w1/mini_batch.shape[0]
        avg_Weight_grad_w2=new_weight_grad_w2/mini_batch.shape[0]
        self.w_1=self.w_1-(avg_Weight_grad_w1*learning_rate)
        self.w_2=self.w_2-(avg_Weight_grad_w2*learning_rate)
        self.round_16=self.round_16-(avg_filter_grad*learning_rate)
        self.round_32=self.round_32-(avg_filter_grad_2*learning_rate)
        return avg_Weight_grad_w1,avg_filter_grad,correct,learning_rate,loss

        
    #test function
    def test(self,test_batch,test_truth_batch):
        correct=0
        for i in range(test_batch.shape[0]):
            cur_sample=test_batch[i:i+1,:,:]
            cur_truth= test_truth_batch[i:i+1]
            sfm_1=self.full_convolution_1(cur_sample)
            pooled_sfm_1,stacked_value_index_list_1=self.pooling(sfm_1)
            sfm_2=self.full_convolution_2(pooled_sfm_1)
            pooled_sfm_2,stacked_value_index_list_2=self.pooling(sfm_2)

            sfm_s,sfm_r,sfm_c=pooled_sfm_2.shape
            flattened_sfm= pooled_sfm_2.reshape(sfm_r*sfm_c*sfm_s)
            dot_prod=np.dot(self.w_1.transpose(),flattened_sfm)
            self.fc_1=np.maximum(dot_prod,dot_prod*0.01)
            dot_prod_2=np.dot(self.w_2.transpose(),self.fc_1)
            self.fc_2=np.maximum(dot_prod_2,dot_prod_2*0.01)
            dot_prod_3=np.dot(self.w_3.transpose(),self.fc_2)
            self.output_layer=np.maximum(dot_prod_3,dot_prod_3*0.01)
            predicted_list=self.softmax()
            predicted=np.argmax(predicted_list)
            predicted_y=np.argmax(cur_truth)

            if predicted==predicted_y:
                correct+=1
        print('test Accuracy:---------------------------------------------')
        return correct/(test_truth_batch.shape[0])


#init class and pre-process dataset
Tester=NeuralNet(X_train,Y_train)
train_mini_batches,truth_mini_batches=Tester.make_mini_batch(1000)
train_mini_batches=np.array(train_mini_batches)

learning_rate=0.6
count=0

#determines num of epochs to run
for u in range(10):
    avg_loss=0
    for i in range(train_mini_batches.shape[0]):
    
        cur_mini_batch= train_mini_batches[i:i+1,:,:,:]
        cur_truth_mini_batches=truth_mini_batches[i:i+1,:]
        
        cur_truth_mini_batches=np.squeeze(cur_truth_mini_batches)
        cur_mini_batch=np.squeeze(cur_mini_batch)
        avg_Weight_grad,avg_filter_grad,correct,learning_rate,loss=Tester.fcl_1(cur_mini_batch,cur_truth_mini_batches,learning_rate)
        avg_loss=avg_loss+loss
        print(correct,learning_rate,loss)
    avg_loss=avg_loss/train_mini_batches.shape[0]
    if count>5:
        learning_rate=learning_rate*0.75
    count+=1
    print("-------------------------------------------------------------")
    print("avg loss=",avg_loss)
    print("-------------------------------------------------------------")


print(Tester.test(X_test,Y_test))