import cv2
import numpy as np
import time
import argparse

# =============================================================================
# Function: Create LOG kernel 
# =============================================================================
def kernel_LOG(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    gauss = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    gauss = gauss/gauss.sum()
    log = gauss*(x**2 + y**2 - 2*sigma**2) / (sigma**4)
    return (sigma**2)*(log - log.mean())

# =============================================================================
# Function: Apply zero padding to the image 
# =============================================================================
def zeroPad(image, padSize):
    iRows, iCols = image.shape
    outputImage = np.zeros((iRows + 2*padSize, iCols+2*padSize), dtype="float32")
    oRows,oCols = outputImage.shape
    for i in range(padSize,oRows - padSize):
        for j in range(padSize,oCols - padSize):
            outputImage[i][j] = image[i-padSize][j-padSize]
    return outputImage

# =============================================================================
# Function: Apply padding to kernel 
# =============================================================================
def kernelPad(kernel,row,col):
    rows,cols = kernel.shape
    padded_kernel = np.zeros(shape=[row,col], dtype ='float32')
    padded_kernel[int((row/2)-(rows/2)):int((row/2)+(rows/2)),int((col/2)-(cols/2)):int((col/2)+(cols/2))] = kernel
    padded_kernel = np.fft.ifftshift(padded_kernel)
    return padded_kernel

# =============================================================================
# Function: Take FFT of image 
# =============================================================================    
def dft2(image):
    rows,cols = image.shape
    fft_rows = np.zeros(shape=[rows,cols], dtype=np.complex)
    for i in range(rows):
        row = image[i]
        fft_rows[i] = np.fft.fft(row)
    fft_cols = np.zeros(shape=[rows,cols], dtype=np.complex)
    for j in range(cols):
        col = fft_rows[:,j]
        fft_cols[:,j] = np.fft.fft(col)
    return fft_cols

# =============================================================================
# Function: Perform convolution in frequency domain 
# =============================================================================    
def convoluteImage(img,kernel):
    rows,cols = img.shape
    result = np.zeros(img.shape,dtype='complex')
    output = np.zeros(img.shape,dtype='complex')
    kernel_pad = kernelPad(kernel,rows,cols)
    img_fft = dft2(img)
    kernel_fft = dft2(kernel_pad)
    result = np.multiply(img_fft,kernel_fft)
    result_conj = np.conj(result)
    output_ifft = dft2(result_conj)
    output = (np.conj(output_ifft))/(rows*cols)
    return output

# =============================================================================
# Read image
# =============================================================================
start = time.time()
image1 = cv2.imread("butterfly.jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
iRows, iCols = image1.shape

# =============================================================================
# Saving copy of image
# =============================================================================

clone = image1.copy()
clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)

# =============================================================================
# Set layer formation parameters
# =============================================================================
layers = 1
scale_layers = 8
sigma = 2
k = 1.4

# =============================================================================
# Scale space creation
# =============================================================================
sigmaScales = np.zeros((1,scale_layers),dtype = 'float32')
scale_space = np.zeros((scale_layers,iRows,iCols),dtype = 'float32')
image1 = image1/255;

for i in range(scale_layers):
    sigmaScales[0][i] = sigma
    sigma = k*sigma

image1 =  image1.astype("float32")
for i in range(len(sigmaScales[0])):
    logKernel = kernel_LOG(2*np.ceil(sigmaScales[0][i]*3)+1,sigmaScales[0][i])
    convOutput = np.square(convoluteImage(image1,logKernel))
    #convOutput = np.square(convOutput)
    scale_space[i,:,:] = convOutput
    
# =============================================================================
# 2d Non-max Suppression
# =============================================================================
max_2d = np.zeros((scale_layers,iRows,iCols),dtype = 'float32')

for i in range(len(sigmaScales[0])):
    octaveImage = scale_space[i,:,:]
    [octR,octC] = octaveImage.shape
    octaveImage = zeroPad(octaveImage,1)
    for j in range(1,octR+1):
        for k in range(1,octC+1):
            #roc = cim[j-1:j+2,k-1:k+2]
            max_2d[i,j-1,k-1] = np.amax(octaveImage[j-1:j+2,k-1:k+2])

# =============================================================================
# 3d Non-max Suppression
# =============================================================================

max_3d = np.zeros((scale_layers,iRows,iCols),dtype = 'float32')
for j in range(1,np.size(max_2d,1)-1):
        for k in range(1,np.size(max_2d,2)-1):
            #roc = max_2d[:,j-1:j+2,k-1:k+2]
            max_3d[:,j,k] = np.amax(max_2d[:,j-1:j+2,k-1:k+2])
max_3d = np.multiply((max_3d == max_2d),max_3d)

# =============================================================================
# Saving location and radius of blobs
# =============================================================================
rowLoc = []
columnLoc = []
radius = [0]*scale_layers
threshold = 0.02
for i in range(np.size(max_3d,0)):
    radius[i] = 1.414*sigmaScales[0][i]
    octImage = scale_space[i,:,:]
    octImage =(octImage == max_3d[i,:,:]) & (octImage>threshold)
    [r,c] = np.where(octImage)
    rowLoc.append(r)
    columnLoc.append(c)

# =============================================================================
# Drawing Circles wherever blobs are detected
# =============================================================================
for i in range(scale_layers):
    for j in range(len(rowLoc[i])):
        cv2.circle(clone,(columnLoc[i][j],rowLoc[i][j]), int(radius[i]) , (0,0,255),2,8)

# =============================================================================
# Calculate Runtime of code
# =============================================================================
end = time.time()
print("Runtime", end - start)

# =============================================================================
# Showing output image with detected blobs
# =============================================================================
cv2.imshow("Clone",clone.astype("uint8"))
cv2.waitKey(0)
cv2.destroyAllWindows()


