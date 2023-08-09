# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def makeSobelFilter(x, y, X_direction, c = 2):
    '''
    Auxiliary function. Aids in sobelFiltering function and applyGradientFilter
    function.
    '''
    indices = []
    for k in range(-1, 2):
        for l in range(-1, 2):
            indices.append([x + k, y + l, 0])
    if X_direction:
        weights = [1, 0, -1, c*1, 0, -c*1, 1, 0, -1]
        for i in range(len(indices)):
            indices[i][2] = weights[i]
    else: 
        weights = [1, c*1, 1, 0, 0, 0, -1, -c*1, -1]
        for i in range(len(indices)):
            indices[i][2] = weights[i]
    return(indices)

def sobelFiltering(img_matrix, direction):
    '''
    Takes a matrix corresponding to a grayscale image (img_matrix) and a boolean
    statement for direction, a True value will apply an X-direction filter (for 
    vertical edges). Applies the Sobel Filter to the image
    
    Borders are handled by truncating the image to the region where the filter
    fits completely.
    
    Returns: a grayscale image matrix after the Sobel filter has been applied.
    '''
    M, N = img_matrix.shape
    new_img = np.zeros((M - 2, N - 2)) #Make a new zero-matrix
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            list_of_indices = makeSobelFilter(i, j, direction) #Get the indices of the pixels inside the filter
            list_of_values = [w*img_matrix[x][y] for [x,y,w] in list_of_indices] #Get the values of said pixels multiplied by the weights
            avg = np.mean(list_of_values) #Get the average of the values
            new_img[i - 1][j- 1] = avg #Write the average to the new image
    return(new_img)




def makeLaplaceFilter(x,y):
    '''
    Auxiliary function. Aids in laplacianFiltering function.
    '''
    indices = []
    for k in range(-1, 2):
        for l in range(-1, 2):
            indices.append([x + k, y + l, 0])
    weights = [0, 1, 0, 1, -4, 1, 0, 1, 0,]
    for i in range(len(indices)):
        indices[i][2] = weights[i]
    return(indices)

def laplacianFiltering(img_matrix):
    '''
    Takes a matrix corresponding to a grayscale image (img_matrix).
    
    Borders are handled by truncating the image to the region where the filter
    fits completely.
    
    Returns: a grayscale image matrix after the Laplacian filter has been applied.
    '''
    M, N = np.shape(img_matrix)
    new_img = np.zeros((M-2,N-2))
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            list_of_indices = makeLaplaceFilter(i, j) #Get the indices of the pixels inside the filter
            list_of_values = [w*img_matrix[x][y] for [x,y,w] in list_of_indices] #Get the values of said pixels multiplied by the weights
            avg = np.mean(list_of_values) #Get the average of the values
            new_img[i - 1][j- 1] = avg #Write the average to the new image
    return(new_img)




def applyGradientFilter(img_matrix, direction, c_val):
    '''
    This function is identical to the function sobelFiltering but it takes an
    additional parameter c_val. It is used in the function edgeDetector.
    '''
    M, N = img_matrix.shape
    new_img = np.zeros((M - 2, N - 2)) #Make a new zero-matrix
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            list_of_indices = makeSobelFilter(i, j, direction, c = c_val) #Get the indices of the pixels inside the filter
            list_of_values = [w*img_matrix[x][y] for [x,y,w] in list_of_indices] #Get the values of said pixels multiplied by the weights
            avg = np.mean(list_of_values) #Get the average of the values
            new_img[i - 1][j- 1] = avg #Write the average to the new image
    return(new_img)


def edgeDetector(img_matrix, c_val, threshold):
    '''
    Takes a matrix corresponding to a grayscale image (img_matrix), an integer
    (either 1 or 2) to decide which gradient filter is used (Prewitt vs Sobel)
    and a float threshold (between 0 and 1).
    
    Borders are handled by truncating the image to the region where the filter
    fits completely.
    
    Returns: a grayscale image matrix highlighting the edges in the original
    image.
    '''
    dX = applyGradientFilter(img_matrix, True, c_val)  # Calculate X derivative
    dY = applyGradientFilter(img_matrix, False, c_val) # Calculate Y derivative
    M, N = dX.shape
    thresholded = np.zeros((M,N))  # Make new 0-matrix where the gradient norm is saved
    gradnorms = np.zeros((M,N))
    
    for i in range(M):
        for j in range(N):
            norm = np.sqrt(dX[i][j]**2 + dY[i][j]**2)   # Calculate gradient magnitude at each point
            if norm > threshold:
                thresholded[i][j] = 1  # Write 1 to 0-matrix only if gradient magnitude is greater than threshold
                gradnorms[i][j] =  norm
                
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(dX, cmap='gray')
    axs[0].set_title('Dx', fontsize=24)
    axs[1].imshow(dY, cmap='gray') 
    axs[1].set_title('Dy', fontsize=24)
    axs[2].imshow(thresholded, cmap='gray') 
    axs[2].set_title('Thresholded edges', fontsize=24)
    
    fig.set_size_inches(12, 6.3)
    fig.tight_layout()
    
    return(dX, dY, thresholded, gradnorms)
    
if __name__ == '__main__':
    # Choose the initial image file and filter type:
    image = "cameraman.jpg"
    img = Image.open(image)
    img_matrix = np.array(img)
    save_as = 'edgeDetector'

    prewittX, prewittY, prewittGrad, prewittNorm = edgeDetector(img_matrix, 1, 20)
    # sobelX, sobelY, sobelGrad = edgeDetector(img_matrix, 2, 20)
    
    
    # fig, axs = plt.subplots(2,3)
    
    # axs[0, 0].imshow(prewittX, cmap='gray') 
    # axs[0, 0].set_title('Dx, c=1', fontsize=24)
    # axs[0, 1].imshow(prewittY, cmap='gray')
    # axs[0, 1].set_title('Dy, c=1', fontsize=24)
    # axs[0, 2].imshow(prewittGrad, cmap='gray')
    # axs[0, 2].set_title('Thresholded Edges, c=1', fontsize=20)
    
    # axs[1, 0].imshow(sobelX, cmap='gray')
    # axs[1, 0].set_title('Dx, c=2', fontsize=24)
    # axs[1, 1].imshow(sobelY, cmap='gray')
    # axs[1, 1].set_title('Dy, c=2', fontsize=24)
    # axs[1, 2].imshow(sobelGrad, cmap='gray')
    # axs[1, 2].set_title('Thresholded Edges, c=2', fontsize=20)
    
    
    # fig.set_size_inches(12, 6.3)
    # fig.tight_layout()
    # fig.savefig(save_as)
    
    #lap = laplacianFiltering(img_matrix)
    
    #plt.imshow(sobeleado, cmap = 'gray')