import numpy as np
# import numpy as np
from numpy.core.shape_base import stack
from tensorflow.keras.datasets import mnist
import time
from tensorflow.keras.utils import to_categorical
import numpy.lib.stride_tricks as nx

#np.shape(sets,rows per set, columns per row)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
Y_train = to_categorical(Y_train) #converts it to one-hpt encoding starting with 0 and ending with 9
Y_test = to_categorical(Y_test)
X_train , X_test = X_train/255.0 , X_test/255.0
# print(Y_train[1])
# print(Y_train.shape)
# print('MNIST Dataset Shape:')
# print('X_train: ' + str(X_train.shape))
# print('Y_train: ' + str(Y_train.shape))
# # print(Y_train[20])
# # input()
# print('X_test:  '  + str(X_test.shape))
# print('Y_test:  '  + str(Y_test.shape))

class NeuralNet():
    def __init__(self,inputs,input_truth,filter_1_dim=(16,3,3),filter_2_dim=(32,16,3,3),max_pool_layers=2,categories=10):
        self.filter_1_dim=filter_1_dim
        self.filter_2_dim=filter_2_dim
        self.input=inputs
        self.input_truth=input_truth
        i_s,i_r,i_c= inputs.shape
        final_connections= i_r//(2**max_pool_layers) * i_c//(2**max_pool_layers) * filter_2_dim[0]
        s,r,c=filter_1_dim
        self.w_1=np.random.rand(final_connections,categories)
        self.round_16=np.random.rand(s,r,c)
        ss,s,r,c=filter_2_dim
        self.round_32=np.random.rand(ss,s,r,c)



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

    
    def Relu(self,input):
        return max(0.01*input,input)

    def softmax(self):
        e_x = np.exp(self.fc_1 - np.max(self.fc_1))
        return e_x / e_x.sum()

    def convolution_1(self,input,filter,stride=1):
        
        i_s,i_r,i_c=input.shape
        f_s,f_r,f_c=filter.shape

        # output_shape=(i_r-(f_r-1))
        input=input.reshape(i_r,i_c)
        filter=filter.reshape(f_r,f_c)
        # print(output_shape)
        input= np.pad(input, 1, mode='constant')

        a=nx.sliding_window_view(input, (f_r,f_c))
        a=a.reshape(a.shape[0]*a.shape[1],a.shape[2],a.shape[3])
        feature_map=np.tensordot(a,filter,axes=2)
        feature_map=feature_map.reshape(i_r,i_c)
        # print(feature_map.shape)
        # time.sleep(5)
        return feature_map

    def convolution_2(self,input,filter,stride=1):
        i_s,i_r,i_c=input.shape
        f_ss,f_s,f_r,f_c=filter.shape

        # output_shape=(i_r-(f_r-1))
        input=input.reshape(i_s,i_r,i_c)
        filter=filter.reshape(f_s,f_r,f_c)
        # print(input.shape,filter.shape,output_shape)
        # input()
        p_width=((0,0),(1,1),(1,1))
        input=np.pad(input,pad_width=p_width,mode='constant',constant_values=0)
        # print(input.shape)

        feature_map=[]

        a=nx.sliding_window_view(input, (f_s,f_r,f_c))
        a=np.squeeze(a)
        
        a=a.reshape(a.shape[0]*a.shape[1],a.shape[2],a.shape[3],a.shape[4])
        feature_map=np.tensordot(a,filter,axes=3)
        feature_map=feature_map.reshape(i_r,i_c)
        # print(feature_map.shape)
        # time.sleep(10)

        # self.feature_map=np.array(feature_map)
        # print(self.feature_map.shape)
        # time.sleep(1)
        return feature_map

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
        # print(total_filters)
        # time.sleep(20)

        for i in range(total_filters):# level 2 filters
            filter=self.round_32[i:i+1,:,:,:]
            featuremap=self.convolution_2(image_data,filter)
            self.stacked_feature_maps_32.append(featuremap)
            # print(len(self.stacked_feature_maps_32))
            # time.sleep(0.5)
        
        self.stacked_feature_maps_32=np.array(self.stacked_feature_maps_32)
        # print(self.stacked_feature_maps_32.shape)
        # time.sleep(20)
        return self.stacked_feature_maps_32

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
        # stacked_value_index_list=np.array(stacked_value_index_list,dtype="i,i")
        # print(stacked_pooled_maps.shape)
        # time.sleep(10)
        stacked_value_index_list=np.array(stacked_value_index_list,dtype="i") 
        # print(stacked_value_index_list.shape)
        # time.sleep(10)
 
        return stacked_pooled_maps,stacked_value_index_list
        


    



    def backprop_for_weights(self,y,flattened_sfm):
        predicted=self.softmax()
        new_gradients=[]
        de_wr_dz=(predicted-y).flatten() #gives error with respect to input of softmax function with one hot encoding and bce. need to find partial deriv of input wr to weight
        for i in range(len(de_wr_dz)):
            error_to_input=de_wr_dz[i]
            row_grad=[]
            for u in range(flattened_sfm.shape[0]):
                temp_grad=error_to_input*flattened_sfm[u] #partial deriv of input wr to weight is just prev node output
                # temp_grad=total_error # get's new w and stores in temp row vector
                row_grad.append(temp_grad)
            new_gradients.append(row_grad) # appends row vector to 2d list to be np-arrayed after
        new_gradients=np.array(new_gradients)
        new_gradients=np.transpose(new_gradients)

        return new_gradients
    
    def backprop_for_flattened_sfm(self,y,flattened_sfm):
        error_per_node_for_flattened_sfm=[]
        predicted=self.softmax()
       
        de_wr_dz=(predicted-y).flatten()
        for i in range(flattened_sfm.shape[0]): # calculates new error per node for flattened layer
            total_error=0
            for u in range(len(de_wr_dz)):
                error_to_input=de_wr_dz[u]
 
                weight_used=self.w_1[i:i+1,u]
                total_error=total_error+(error_to_input*weight_used)
            error_per_node_for_flattened_sfm.append(total_error)
   
        error_per_node_for_flattened_sfm=np.array(error_per_node_for_flattened_sfm)
   

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
            # print(ref_list,ref_list.shape)
            # print(f_c)
            # time.sleep(10)
            # ref_list=ref_list.reshape((f_c//2)**2)
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
        # print(stacked_value_index_list,"herehere")
        # time.sleep(20)

        f_s,f_r,f_c= stacked_feature_map_dim
        stacked_error_feature_maps=[]
        # error_per_node_for_conv2_layer.reshape(f_s,f_r//2,f_c//2)
        error_per_node_for_conv2_layer=error_per_node_for_conv2_layer.reshape(f_s*f_r//2*f_c//2)
        # print(error_per_node_for_conv2_layer.shape,'here2')
        # time.sleep(20)
        counter=0
        for i in range(f_s):
            error_feature_map=np.zeros((f_r, f_c))
            ref_list=stacked_value_index_list[i:i+1,:]
            ref_list=ref_list.reshape(((f_c//2)**2),2)
            for location in range((f_c//2)**2):
                ref=ref_list[location]
                # print(ref,type(ref[0]),type(ref[1]),counter,error_feature_map.shape)
                # print(error_per_node_for_conv2_layer.shape)
                # time.sleep(5)
                # error_feature_map[ref[0],ref[1]]=error_per_node_for_conv2_layer[counter]
                error_feature_map[ref[0]][ref[1]]=error_per_node_for_conv2_layer[counter]
                counter+=1
            stacked_error_feature_maps.append(error_feature_map)

        stacked_error_feature_maps=np.array(stacked_error_feature_maps).reshape(f_s,f_r,f_c)
        return stacked_error_feature_maps


    def backprop_filter_2(self,stacked_error_feature_maps,input):
        # count=0

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
                        # print(output)
                        # input()
                        # print(output,count,ss,r,c,filter_ss)
                        # count=count+1
                        # time.sleep(1)
        
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
        # e_s,e_r,e_c=mod_stacked_error_feature_maps.shape #(32,14,17)
        f_ss,f_s,f_r,f_c=temp_kernel.shape #filter (32,16,3,17)
        #run through all 16 filter layers over 1 single layer of mod_stacked. then repeat 32 times and average it out
        final_list=[]
        for i in range(f_ss):
            input=mod_stacked_error_feature_maps[i:i+1,:,:]
            input=input.flatten()
            #remove last 2 pixels here maybe?\
            input=np.delete(input,[input.shape[0]-1,input.shape[0]-2])
            temp_list=[]
            for u in range(f_s):
                kernel=temp_kernel[i:i+1,u:u+1,:,:]
                kernel=kernel.flatten()
                kernel=np.insert(kernel,3,np.zeros(13))
                kernel=np.insert(kernel,19,np.zeros(13))
                # print(kernel)
                # time.sleep(10)

                test=np.convolve(input,kernel,'full')
                # test=np.delete(test,[256,257])
                test=test.reshape(16,16)
                temp_list.append(test)
            temp_list=np.array(temp_list)
            final_list.append(temp_list)
        final_list=np.array(final_list)

        final_list=final_list[:,:,1:-1,1:-1]

        final_list=np.mean(final_list,axis=0)


        return final_list
        

    def fcl_1(self,mini_batch,truth_mini_batch,learning_rate=0.005):
    
        new_weight_grad=np.zeros(self.w_1.shape)
        new_filter_grad=np.zeros(self.filter_1_dim)
        new_filter_grad_2=np.zeros(self.filter_2_dim)
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
            # print(sfm_1.shape,pooled_sfm_1.shape,sfm_2.shape,pooled_sfm_2.shape)
            # time.sleep(20)
            sfm_s,sfm_r,sfm_c=pooled_sfm_2.shape
            flattened_sfm= pooled_sfm_2.reshape(sfm_r*sfm_c*sfm_s)
        
            self.fc_1=np.maximum(np.dot(self.w_1.transpose(),flattened_sfm),0)
            predicted_list=self.softmax()
            predicted=np.argmax(predicted_list)
            predicted_y=np.argmax(cur_truth)
            # print(predicted_list)
     
            loss=loss+(predicted_list[predicted_y]-cur_truth[predicted_y])
           
        # done till here
            if predicted==predicted_y:
                correct+=1
            # print(new_weight_grad.shape,self.backprop_for_weights(cur_truth,flattened_sfm).shape)
            new_weight_grad=new_weight_grad+self.backprop_for_weights(cur_truth,flattened_sfm)
            error_per_node_for_flattened_sfm=self.backprop_for_flattened_sfm(cur_truth,flattened_sfm)
            stacked_error_feature_maps=self.backprop_maxpool_2(sfm_2.shape,error_per_node_for_flattened_sfm,stacked_value_index_list_2)
            # print(stacked_error_feature_maps.shape)
            # time.sleep(20)
            new_filter_grad_2=new_filter_grad_2+self.backprop_filter_2(stacked_error_feature_maps,pooled_sfm_1)
            backpropped_error=self.error_for_input_conv2(stacked_error_feature_maps)
            stacked_error_feature_maps=self.backprop_maxpool_1(sfm_1.shape,backpropped_error,stacked_value_index_list_1)
            new_filter_grad=new_filter_grad+self.backprop_filter_1(stacked_error_feature_maps,cur_sample)
           
        loss=loss/mini_batch.shape[0]
        avg_filter_grad=new_filter_grad/mini_batch.shape[0]
        avg_filter_grad_2=new_filter_grad_2/mini_batch.shape[0]
        avg_Weight_grad=new_weight_grad/mini_batch.shape[0]
        self.w_1=self.w_1-(avg_Weight_grad*learning_rate)
        self.round_16=self.round_16-(avg_filter_grad*learning_rate)
        self.round_32=self.round_32-(avg_filter_grad_2*learning_rate)
        return avg_Weight_grad,avg_filter_grad,correct,learning_rate,loss

        
    def test(self,test_batch,test_truth_batch):
        correct=0
        for i in range(test_batch.shape[0]):
            cur_sample=test_batch[i:i+1,:,:]
            cur_truth= test_truth_batch[i:i+1]
            sfm_1=self.full_convolution_1(cur_sample)
            pooled_sfm_1,stacked_value_index_list_1=self.pooling(sfm_1)
            sfm_2=self.full_convolution_2(pooled_sfm_1)
            pooled_sfm_2,stacked_value_index_list_2=self.pooling(sfm_2)
            # print(sfm_1.shape,pooled_sfm_1.shape,sfm_2.shape,pooled_sfm_2.shape)
            # time.sleep(20)
            sfm_s,sfm_r,sfm_c=pooled_sfm_2.shape
            flattened_sfm= pooled_sfm_2.reshape(sfm_r*sfm_c*sfm_s)
        
            self.fc_1=np.maximum(np.dot(self.w_1.transpose(),flattened_sfm),0)
            predicted_list=self.softmax()
            predicted=np.argmax(predicted_list)
            predicted_y=np.argmax(cur_truth)

            if predicted==predicted_y:
                correct+=1
        print('test Accuracy:---------------------------------------------')
        return correct/(test_truth_batch.shape[0])


    
Tester=NeuralNet(X_train,Y_train)
train_mini_batches,truth_mini_batches=Tester.make_mini_batch(6000)
train_mini_batches=np.array(train_mini_batches)
learning_rate=0.005
count=0


for i in range(10):
    for i in range(train_mini_batches.shape[0]):
        if count>5:
            learning_rate=(learning_rate*0.99)
            count=0
        cur_mini_batch= train_mini_batches[i:i+1,:,:,:]
        cur_truth_mini_batches=truth_mini_batches[i:i+1,:]

        cur_truth_mini_batches=cur_truth_mini_batches.reshape(10,10)
        cur_mini_batch=cur_mini_batch.reshape(10,28,28)

        avg_Weight_grad,avg_filter_grad,correct,learning_rate,loss=Tester.fcl_1(cur_mini_batch,cur_truth_mini_batches,learning_rate)
        print(correct,learning_rate,loss)
        count+=1


    print(Tester.test(X_test,Y_test))
