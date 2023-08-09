# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def makeFilter(x, y, filter_size, gaussian=False):
    '''
    Funcion auxiliar para las siguientes funciones. Devuelve los indices
    correspondientes a los pixeles que abarca el filtro.
    '''
    indices = []
    for k in range(-filter_size, filter_size+1):
        for l in range(-filter_size, filter_size+1):
            if gaussian:
                weight = np.exp(-(k**2/float(filter_size)+l**2/float(filter_size)))
            else:
                weight = 1
            indices.append((x + k, y + l, weight))
    return(indices)


def medianFiltering(img_matrix, filter_size):
    '''
    Takes a matrix corresponding to a grayscale image (img_matrix) and a filter
    size. filter_size should be an integer corresponding to the 'radius' of
    the filter. Example: if the user wants to apply a 5x5 filter, they should
    use filter_size = 2.
    
    Borders are handled by truncating the image to the region where the filter
    fits completely.
    
    Returns: a smoothed image using an MEDIAN filter.
    '''
    M, N = img_matrix.shape
    new_img = np.zeros((M - 2*filter_size, N - 2*filter_size)) #Make a new zero-matrix
    for i in range(filter_size, M - filter_size):
        for j in range(filter_size, N - filter_size):
            list_of_indices = makeFilter(i,j,filter_size) #Get the indices of the pixels inside the filter
            list_of_values = [w*img_matrix[x][y] for (x,y,w) in list_of_indices] #Get the values of said pixels multiplied by the weights
            median = np.median(list_of_values) #Get the median of the values
            new_img[i - filter_size][j- filter_size] = median #Write the median to the new image
    return(new_img)


def averageFiltering(img_matrix, filter_size):
    '''
    Takes a matrix corresponding to a grayscale image (img_matrix) and a filter
    size. filter_size should be an integer corresponding to the 'radius' of
    the filter. Example: if the user wants to apply a 5x5 filter, they should
    use filter_size = 2.
    
    Borders are handled by truncating the image to the region where the filter
    fits completely.
    
    Returns: a smoothed image using an AVERAGING filter.
    '''
    M, N = img_matrix.shape
    new_img = np.zeros((M - 2*filter_size, N - 2*filter_size)) #Make a new zero-matrix
    for i in range(filter_size, M - filter_size):
        for j in range(filter_size, N - filter_size):
            list_of_indices = makeFilter(i,j,filter_size) #Get the indices of the pixels inside the filter
            list_of_values = [w*img_matrix[x][y] for (x,y,w) in list_of_indices] #Get the values of said pixels multiplied by the weights
            average = np.mean(list_of_values) #Get the average of the values
            new_img[i - filter_size][j- filter_size] = average #Write the average to the new image
    return(new_img)


def weightavgFiltering(img_matrix, filter_size):
    '''
    Takes a matrix corresponding to a grayscale image (img_matrix) and a filter
    size. filter_size should be an integer corresponding to the 'radius' of
    the filter. Example: if the user wants to apply a 5x5 filter, they should
    use filter_size = 2.
    
    Borders are handled by truncating the image to the region where the filter
    fits completely.
    
    Returns: a smoothed image using a GAUSSIAN AVERAGE filter.
    '''
    M, N = img_matrix.shape
    new_img = np.zeros((M - 2*filter_size, N - 2*filter_size)) #Make a new zero-matrix
    for i in range(filter_size, M - filter_size):
        for j in range(filter_size, N - filter_size):
            list_of_indices = makeFilter(i, j, filter_size, gaussian=True) #Get the indices of the pixels inside the filter
            list_of_values = [w*img_matrix[x][y] for (x,y,w) in list_of_indices] #Get the values of said pixels multiplied by the weights
            average = np.mean(list_of_values) #Get the weighted average of the values
            new_img[i - filter_size][j- filter_size] = average #Write the weighted average to the new image
    return(new_img)


def avgcolorFiltering(img_matrix, filter_size):
    '''
    Takes a matrix corresponding to an RGB image (img_matrix) and a filter
    size. filter_size should be an integer corresponding to the 'radius' of
    the filter. Example: if the user wants to apply a 5x5 filter, they should
    use filter_size = 2.
    
    Borders are handled by truncating the image to the region where the filter
    fits completely.
    
    Returns: a smoothed GRAYSCALE image using an AVERAGING filter.
    '''
    M, N, D = img_matrix.shape
    new_img = np.zeros((M - 2*filter_size, N - 2*filter_size)) #Make a new zero-matrix
    for i in range(filter_size, M - filter_size):
        for j in range(filter_size, N - filter_size):
            list_of_indices = makeFilter(i, j, filter_size) #Get the indices of the pixels inside the filter
            list_of_values = []
            for index in list_of_indices:
                x = index[0]
                y = index[1]
                pixel = img_matrix[x][y]
                for value in pixel:
                    list_of_values.append(value)
            average = np.mean(list_of_values) #Get the weighted average of the values
            new_img[i - filter_size][j- filter_size] = average #Write the weighted average to the new image
    return(new_img)


def saveImgs(image_file, filtering, fs1, fs2, fs3, save_as):
    '''
    Auxiliary function used to plot and save the generated images.
    '''
    pic = Image.open(image_file)
    img = np.array(pic)
    
    I1 = averageFiltering(img, fs1)
    I2 = averageFiltering(img, fs2)
    I3 = averageFiltering(img, fs3)
    
    title1 = 'filter_size = ' + str(fs1)
    title2 = 'filter_size = ' + str(fs2)
    title3 = 'filter_size = ' + str(fs3)
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original', fontsize=32)
    axs[0, 1].imshow(I1, cmap='gray')
    axs[0, 1].set_title(title1, fontsize=32)
    axs[1, 0].imshow(I2, cmap='gray')
    axs[1, 0].set_title(title2, fontsize=32)
    axs[1, 1].imshow(I3, cmap='gray')
    axs[1, 1].set_title(title3, fontsize=32)
    fig.set_size_inches(16, 12)
    fig.tight_layout()
    fig.savefig(save_as)