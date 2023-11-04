
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

#Hooking to obtain the output from layer
def hooking(model,layer,img):
    outputs = {}
    def hook(module,input,output):
        outputs[layer] = output
    if "." in layer:
       splitting = layer.split('.')
       layer_name = model
       for i in splitting:
            
            layer_name = getattr(layer_name,i)
       layer = layer_name
           
    else:
       layer_name = layer
       layer = getattr(model,layer_name)
    layer.register_forward_hook(hook)
    with torch.no_grad():
        model(img)
    output_tensor = outputs[layer][0]
    return output_tensor
                
            
#Splitting the model into conv2d layers            
def layer_split(model):
    
    layers = []
    print("The model Conv layers are as follows: ")
    for name, module in model.named_modules():
        if isinstance(module,nn.Conv2d):
           
           layers.append(name)
           print("The layer name: {0}; The layer: {1}".format(name,module))
    
    return layers

#Visualizing a particular layer's output -- the actual function
def visualizer(model,image):
    
    layers = layer_split(model)
    print("\n")
    print("\033[1m" + "Please use the above printed layers as reference for selection")
    print("\n")
    flag_1 = True
    
    while flag_1:
        layer_name = str(input("Enter name of the layer you wish to visualize: "))
        if layer_name in layers:
            flag_1 = False
        else:
            print("Invalid layer name; choose again")
    
    
    print("Layer: ",layer_name)
        
    output = hooking(model,layer_name,image)
    output_img = output.permute(1,2,0).cpu().numpy()
    #output_img = np.sum(output_img,axis=2)
    print("Number of feature maps in the layer: ",output_img.shape[2])
    sum_list= []
    for i in range(output_img.shape[2]):
       sum_values = output_img[:,:,i].sum()
       
       sum_list.append(sum_values.item())
    selected_maps = np.flip(np.argsort(sum_list)[-5:])
    print("Top 5 feature maps: ",selected_maps)
    
    
    for i in selected_maps:
        print("\n")
        print("Map index: ",i)
        plt.imshow(output_img[:,:,i])
        plt.colorbar()
        plt.show()
        
    
 