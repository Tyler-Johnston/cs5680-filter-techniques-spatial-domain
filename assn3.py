import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.signal import find_peaks

circuitIm = cv2.imread('Circuit.jpg', cv2.IMREAD_GRAYSCALE)
moonIm = cv2.imread('Moon.jpg', cv2.IMREAD_GRAYSCALE)
riceIm = cv2.imread('Rice.jpg', cv2.IMREAD_GRAYSCALE)

# the gif formatted images would register as NoneType using cv2.imread
textIm = cv2.cvtColor(mpimg.imread('Text.gif'), cv2.COLOR_RGBA2GRAY) # this image had 4 channels. this converts it to grayscale format instead of RGB
text1Im = mpimg.imread('Text1.gif')

# PROBLEM 1 QUESTION 1
def AverageFiltering(im, mask):
    """
    Inputs:
        - im: np.array, the original grayscale image.
        - mask: np.array, the square filter with an odd number of rows and columns.
    Outputs:
        - processedImage: np.array, the low-pass average filtered image.
    """
    maskHeight, maskWidth = mask.shape
    imageHeight, imageWidth = im.shape

    # validate the mask
    if (maskHeight != maskWidth) or (maskHeight % 2 == 0) or (maskWidth % 2 == 0):
        raise ValueError("Mask must be a square with an odd number of rows and columns")

    total = 0
    for i in range(maskHeight):
        for j in range(maskWidth):
            if mask[i][j] <= 0:
                raise ValueError("All elements in the mask must be positive")
            total += mask[i][j]
    
    # change to uint8 to round it down from 1.0000000...2 to 1
    total = np.uint8(total)
    if total != 1:
        raise ValueError("The sum of all elements in the mask must be 1")
    
    # check symmetry around the center
    center = maskHeight // 2
    for i in range(center):
        for j in range(maskWidth):
            if mask[i, j] != mask[maskHeight-i-1, j] or mask[j, i] != mask[j, maskWidth-i-1]:
                raise ValueError("Elements of the mask must be symmetric around the center")
    
    # prepare the output image by initializing a np array the same size as 'im' but filled with zeros
    processedImage = np.zeros_like(im, dtype=np.uint8)
    
    # padding the image
    padSize = maskHeight // 2
    padImage = np.pad(im, ((padSize, padSize), (padSize, padSize)), mode='constant')
    
    # perform convolution operation
    for i in range(imageHeight):
        for j in range(imageWidth):
            subMatrix = padImage[i:i+maskHeight, j:j+maskWidth]
            processedImage[i, j] = np.sum(subMatrix * mask)

    return processedImage

# define standard 5x5 mask and weighted standard 3x3 mask
averageStandard5x5Mask = np.ones((5,5)) / 25
averageWeighted3x3Mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

# apply the average filtering with these masks
averageStandard5x5Image = AverageFiltering(circuitIm, averageStandard5x5Mask)
averageWeighted3x3Image = AverageFiltering(circuitIm, averageWeighted3x3Mask)

# plotting
plt.figure(figsize=(10, 5)) # Figure 1
plt.suptitle("Problem 1 Question 1 - AverageFiltering() for Circuit.jpg")
plt.subplot(1, 3, 1)
plt.imshow(circuitIm, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(averageStandard5x5Image, cmap='gray')
plt.title("5x5 Standard Mask")

plt.subplot(1, 3, 3)
plt.imshow(averageWeighted3x3Image, cmap='gray')
plt.title("3x3 Weighted Mask")
plt.tight_layout()

# PROBLEM 1 QUESTION 2:
def MedianFiltering(im, mask):
    """
    Inputs:
        - im: np.array, the original grayscale image.
        - mask: np.array, the square filter with an odd number of rows and columns.
    Outputs:
        - processedImage: np.array, the median filtered image.
    """
    maskHeight, maskWidth = mask.shape
    imageHeight, imageWidth = im.shape

    # validate the mask
    if (maskHeight != maskWidth) or (maskHeight % 2 == 0) or (maskWidth % 2 == 0):
        raise ValueError("Mask must be a square with an odd number of rows and columns")

    total = 0
    for i in range(maskHeight):
        for j in range(maskWidth):
            if mask[i][j] <= 0:
                raise ValueError("All elements in the mask must be positive")
            total += mask[i][j]
    
    # prepare the output image by initializing a np array the same size as 'im' but filled with zeros
    processedImage = np.zeros_like(im, dtype=np.uint8)
    
    # padding the image
    padSize = maskHeight // 2
    padImage = np.pad(im, ((padSize, padSize), (padSize, padSize)), mode='constant')
    
    # perform median filtering
    for i in range(imageHeight):
        for j in range(imageWidth):
            subMatrix = padImage[i:i+maskHeight, j:j+maskWidth]
            values = []
            for k in range(maskHeight):
                for l in range(maskWidth):
                    values.extend([subMatrix[k, l]] * mask[k, l])
            processedImage[i, j] = np.median(values)

    return processedImage

# define the standard 3x3 mask and the weighted 3x3 mask
medianStandard3x3Mask = np.ones((3, 3), dtype=int)
medianWeighted3x3Mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

# apply the median filtering with these masks
medianStandard3x3Image = MedianFiltering(circuitIm, medianStandard3x3Mask)
medianWeighted3x3Image = MedianFiltering(circuitIm, medianWeighted3x3Mask)

# plotting
plt.figure(figsize=(10, 5)) # Figure 2
plt.suptitle("Problem 1 Question 2 - MedianFiltering() for Circuit.jpg")
plt.subplot(1, 3, 1)
plt.imshow(circuitIm, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(medianStandard3x3Image, cmap='gray')
plt.title("5x5 Standard Mask")

plt.subplot(1, 3, 3)
plt.imshow(medianWeighted3x3Image, cmap='gray')
plt.title("3x3 Weighted Mask")
plt.tight_layout()

# PROBLEM 1 QUESTION 3
laplacianFilter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
filteredImage = cv2.filter2D(np.double(moonIm), -1, laplacianFilter)
clippedFilteredImage = filteredImage.clip(0, 255).astype(np.uint8)
scaledFilteredImage = ((filteredImage - filteredImage.min()) / (filteredImage.max() - filteredImage.min()) * 255).astype(np.uint8)
enhancedImage = cv2.add(np.double(moonIm), filteredImage).clip(0, 255).astype(np.uint8)

# plotting
plt.figure(figsize=(12, 6)) # Figure 3
plt.suptitle("Problem 1 Question 3 - Laplacian Mask for Moon.jpg")
plt.subplot(1, 4, 1)
plt.imshow(moonIm, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(clippedFilteredImage, cmap='gray')
plt.title("Filtered Image")

plt.subplot(1, 4, 3)
plt.imshow(scaledFilteredImage, cmap='gray')
plt.title("Scaled Filtered Image")

plt.subplot(1, 4, 4)
plt.imshow(enhancedImage, cmap='gray')
plt.title("Enhanced Image")
plt.tight_layout()

# PROBLEM 2
def FindEdgeInfo(im, bin, thresholdingValue=80):
    """
    Inputs:
        im (np.array): Grayscale image
        bin (int): Number of bins for the histogram
        
    Outputs:
        edges (np.array): Binary image containing important edges
        edgeHist (np.array): Edge histogram containing counts of the orientation of edges in each bin
    """
    im = np.double(im)
    # represents horizontal changes in intensity, using horizontal edge detector
    Gx = cv2.filter2D(im, -1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    # represents verical changes in intensity , using vertical edge detector
    Gy = cv2.filter2D(im, -1, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    # is a 2D numpy array representing the intensities of edge locations
    magnitude = np.sqrt(Gx**2 + Gy**2)
    # is a 2D numpy array that contains angles for each pixel in the processed image from [-pi, pi]
    direction = np.arctan2(Gy, Gx)
    direction = np.degrees(direction) % 360  # converts [-pi, pi] to [0, 360]

    # threshold on the magnitude to retain important edges while avoiding excessive noise
    edges = (magnitude > np.percentile(magnitude, thresholdingValue)).astype(np.uint8)

    # create a histogram with 'bin' number of bins
    # this should count the number of angles in the 'direction' numpy array and assign then to their appropriate bins
    histogram = [0] * bin
    anglesPerBin = 360 / bin
    directionHeight, directionWidth = direction.shape
    for i in range(directionHeight):
        for j in range(directionWidth):
            # make sure any potential 'NaN' is not being added to the histogram
            if not np.isnan(direction[i][j]):
                # get the index of the necessary bin for each angle in the image
                # adding '% bin' in the event of index is 30, so it will roll over to bin 0. this makes sense as it completed one cycle
                index = np.trunc(direction[i][j] / anglesPerBin).astype(np.uint8) % bin
                histogram[index] += 1

    return edges, histogram

binCount = 30
edges, edgeHist = FindEdgeInfo(riceIm, binCount)

# plotting
plt.figure(figsize=(15,5)) # Figure 4
plt.suptitle("Problem 2 - FindEdgeInfo() for Rice.jpg")
plt.subplot(1, 3, 1)
plt.imshow(riceIm, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edges')

plt.subplot(1, 3, 3)
plt.bar(range(binCount), edgeHist)
plt.title('Histogram')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.tight_layout()

# PROBLEM 3
def RemoveStripes(im):
    histogram = [0] * 256
    imageHeight, imageWidth = im.shape
    whitePixelValue = 250

    # get the histogram for the original image
    for i in range(imageHeight):
        for j in range(imageWidth):
            index = im[i, j]
            histogram[index] += 1

    # get peaks with the lowest and highest count in the histogram
    peaks, _ = find_peaks(histogram, prominence=10)
    if len(peaks) >= 2:
        lowerThreshold = peaks[0]
        lowerHistogramCount = histogram[lowerThreshold]
        for i in range(1, len(peaks)):
            if histogram[peaks[i]] < lowerHistogramCount:
                lowerThreshold = peaks[i]
                lowerHistogramCount = histogram[lowerThreshold]
        
        upperThreshold = peaks[0]
        upperHistogramCount = histogram[upperThreshold]
        for i in range(1, len(peaks)):
            if histogram[peaks[i]] > upperHistogramCount:
                upperThreshold = peaks[i]
                upperHistogramCount = histogram[upperThreshold]

    else:
        # use default thresholds if somehow there aren't 2 peaks
        lowerThreshold = imageHeight // 3
        upperThreshold = np.median(im)

    fixedImage = np.copy(im)
    for i in range(imageHeight):
        for j in range(imageWidth):
            if lowerThreshold < im[i, j] < upperThreshold:
                fixedImage[i, j] = whitePixelValue
    
    return fixedImage, histogram

fixedTextIm, histogramTextIm = RemoveStripes(textIm)
fixedText1Im, histogramText1Im = RemoveStripes(text1Im)

plt.figure(figsize=(15,5)) # Figure 5
plt.suptitle("Problem 3 Part 1 - RemoveStripes() for Text.gif")
plt.subplot(1, 3, 1)
plt.imshow(textIm, cmap='gray')
plt.title('Original Text.gif Image')

plt.subplot(1, 3, 2)
plt.imshow(fixedTextIm, cmap='gray')
plt.title('Fixed Text.gif Image')

plt.subplot(1, 3, 3)
plt.bar(range(256), histogramTextIm)
plt.title('Text.gif Histogram')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.tight_layout()

plt.figure(figsize=(15,5)) # Figure 6
plt.suptitle("Problem 3 Part 2 - RemoveStripes() for Text1.gif")
plt.subplot(1, 3, 1)
plt.imshow(text1Im, cmap='gray')
plt.title('Original Text1.gif Image')

plt.subplot(1, 3, 2)
plt.imshow(fixedText1Im, cmap='gray')
plt.title('Fixed Text1.gif Image')

plt.subplot(1, 3, 3)
plt.bar(range(256), histogramText1Im)
plt.title('Text1.gif Histogram')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.tight_layout()

plt.show()