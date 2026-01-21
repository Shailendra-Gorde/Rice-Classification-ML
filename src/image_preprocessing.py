"""
Rice Classification - Image Feature Extraction Module

This module extracts morphological and visual features from rice grain images:
- Color features (mean RGB, HSV)
- Texture features (GLCM, Local Binary Pattern)
- Size and Shape features (Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea)
"""

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops
from scipy.spatial.distance import euclidean
import os


class RiceImageFeatureExtractor:
    """
    A class to extract features from rice grain images.
    """

    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = [
            # Color features
            'Color_Mean_R', 'Color_Mean_G', 'Color_Mean_B',
            'Color_Mean_H', 'Color_Mean_S', 'Color_Mean_V',
            # Texture features (GLCM)
            'Texture_Contrast', 'Texture_Dissimilarity', 'Texture_Homogeneity', 'Texture_Energy',
            # Texture features (LBP)
            'Texture_LBP_Mean', 'Texture_LBP_Std',
            # Size features
            'Size_Area', 'Size_Perimeter',
            # Shape features
            'Shape_MajorAxisLength', 'Shape_MinorAxisLength',
            'Shape_ConvexArea', 'Shape_Eccentricity',
            'Shape_Extent', 'Shape_Roundness', 'Shape_AspectRatio',
            # Additional shape features
            'Shape_EquivDiameter', 'Shape_Solidity'
        ]

    def preprocess_image(self, image_path):
        """
        Load and preprocess the rice grain image.

        Parameters:
        -----------
        image_path : str or np.ndarray
            Path to image file or image array

        Returns:
        --------
        tuple
            (original_image, gray_image, binary_mask)
        """
        # Load image
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from: {image_path}")
        else:
            image = image_path.copy()

        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Convert to grayscale
        if len(image_rgb.shape) == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_rgb

        # Create binary mask (threshold to separate rice grain from background)
        # Using Otsu's method for automatic thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (assuming rice is darker than background)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        return image_rgb, gray, binary

    def extract_color_features(self, image_rgb):
        """
        Extract color features from RGB image.

        Parameters:
        -----------
        image_rgb : np.ndarray
            RGB image

        Returns:
        --------
        dict
            Dictionary of color features
        """
        # Mean RGB values
        mean_r = np.mean(image_rgb[:, :, 0])
        mean_g = np.mean(image_rgb[:, :, 1])
        mean_b = np.mean(image_rgb[:, :, 2])

        # Convert to HSV
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # Mean HSV values
        mean_h = np.mean(image_hsv[:, :, 0])
        mean_s = np.mean(image_hsv[:, :, 1])
        mean_v = np.mean(image_hsv[:, :, 2])

        return {
            'Color_Mean_R': float(mean_r),
            'Color_Mean_G': float(mean_g),
            'Color_Mean_B': float(mean_b),
            'Color_Mean_H': float(mean_h),
            'Color_Mean_S': float(mean_s),
            'Color_Mean_V': float(mean_v)
        }

    def extract_texture_features_glcm(self, gray_image, binary_mask):
        """
        Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).

        Parameters:
        -----------
        gray_image : np.ndarray
            Grayscale image
        binary_mask : np.ndarray
            Binary mask of rice grain

        Returns:
        --------
        dict
            Dictionary of GLCM texture features
        """
        # Apply mask to extract only rice grain region
        masked_gray = gray_image * (binary_mask > 0)

        # Ensure we have valid region
        if np.sum(binary_mask > 0) < 100:  # Too small region
            return {
                'Texture_Contrast': 0.0,
                'Texture_Dissimilarity': 0.0,
                'Texture_Homogeneity': 1.0,
                'Texture_Energy': 0.0
            }

        # Normalize to 0-255
        if masked_gray.max() > 0:
            masked_gray = (masked_gray * 255 / masked_gray.max()).astype(np.uint8)
        else:
            masked_gray = masked_gray.astype(np.uint8)

        # Calculate GLCM
        try:
            glcm = graycomatrix(
                masked_gray,
                distances=[1],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                symmetric=True,
                normed=True
            )

            # Extract properties
            contrast = np.mean(graycoprops(glcm, 'contrast'))
            dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
            homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
            energy = np.mean(graycoprops(glcm, 'energy'))

        except Exception as e:
            # Return default values if GLCM calculation fails
            contrast = 0.0
            dissimilarity = 0.0
            homogeneity = 1.0
            energy = 0.0

        return {
            'Texture_Contrast': float(contrast),
            'Texture_Dissimilarity': float(dissimilarity),
            'Texture_Homogeneity': float(homogeneity),
            'Texture_Energy': float(energy)
        }

    def extract_texture_features_lbp(self, gray_image, binary_mask):
        """
        Extract texture features using Local Binary Pattern (LBP).

        Parameters:
        -----------
        gray_image : np.ndarray
            Grayscale image
        binary_mask : np.ndarray
            Binary mask of rice grain

        Returns:
        --------
        dict
            Dictionary of LBP texture features
        """
        # Apply mask
        masked_gray = gray_image * (binary_mask > 0)

        # Normalize
        if masked_gray.max() > 0:
            masked_gray = (masked_gray * 255 / masked_gray.max()).astype(np.uint8)
        else:
            masked_gray = masked_gray.astype(np.uint8)

        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(masked_gray, n_points, radius, method='uniform')

        # Mask out background
        lbp_masked = lbp[binary_mask > 0]

        if len(lbp_masked) > 0:
            lbp_mean = np.mean(lbp_masked)
            lbp_std = np.std(lbp_masked)
        else:
            lbp_mean = 0.0
            lbp_std = 0.0

        return {
            'Texture_LBP_Mean': float(lbp_mean),
            'Texture_LBP_Std': float(lbp_std)
        }

    def extract_size_shape_features(self, binary_mask):
        """
        Extract size and shape features from binary mask.

        Parameters:
        -----------
        binary_mask : np.ndarray
            Binary mask of rice grain

        Returns:
        --------
        dict
            Dictionary of size and shape features
        """
        # Find contours
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # Return default values if no contour found
            return {
                'Size_Area': 0.0,
                'Size_Perimeter': 0.0,
                'Shape_MajorAxisLength': 0.0,
                'Shape_MinorAxisLength': 0.0,
                'Shape_ConvexArea': 0.0,
                'Shape_Eccentricity': 0.0,
                'Shape_Extent': 0.0,
                'Shape_Roundness': 0.0,
                'Shape_AspectRatio': 1.0,
                'Shape_EquivDiameter': 0.0,
                'Shape_Solidity': 0.0
            }

        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Basic size features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        if area == 0:
            return {
                'Size_Area': 0.0,
                'Size_Perimeter': 0.0,
                'Shape_MajorAxisLength': 0.0,
                'Shape_MinorAxisLength': 0.0,
                'Shape_ConvexArea': 0.0,
                'Shape_Eccentricity': 0.0,
                'Shape_Extent': 0.0,
                'Shape_Roundness': 0.0,
                'Shape_AspectRatio': 1.0,
                'Shape_EquivDiameter': 0.0,
                'Shape_Solidity': 0.0
            }

        # Fit ellipse to get major and minor axis
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2)) if major_axis > 0 else 0
        else:
            # Approximate from bounding box
            rect = cv2.minAreaRect(largest_contour)
            major_axis = max(rect[1])
            minor_axis = min(rect[1])
            eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2)) if major_axis > 0 else 0

        # Convex hull
        hull = cv2.convexHull(largest_contour)
        convex_area = cv2.contourArea(hull)

        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        bounding_box_area = w * h
        extent = area / bounding_box_area if bounding_box_area > 0 else 0

        # Roundness (circularity)
        if perimeter > 0:
            roundness = (4 * np.pi * area) / (perimeter ** 2)
        else:
            roundness = 0

        # Aspect ratio
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0

        # Equivalent diameter
        equiv_diameter = 2 * np.sqrt(area / np.pi)

        # Solidity (ratio of contour area to convex hull area)
        solidity = area / convex_area if convex_area > 0 else 0

        return {
            'Size_Area': float(area),
            'Size_Perimeter': float(perimeter),
            'Shape_MajorAxisLength': float(major_axis),
            'Shape_MinorAxisLength': float(minor_axis),
            'Shape_ConvexArea': float(convex_area),
            'Shape_Eccentricity': float(eccentricity),
            'Shape_Extent': float(extent),
            'Shape_Roundness': float(roundness),
            'Shape_AspectRatio': float(aspect_ratio),
            'Shape_EquivDiameter': float(equiv_diameter),
            'Shape_Solidity': float(solidity)
        }

    def extract_all_features(self, image_path):
        """
        Extract all features from a rice grain image.

        Parameters:
        -----------
        image_path : str or np.ndarray
            Path to image file or image array

        Returns:
        --------
        dict
            Dictionary containing all extracted features
        """
        # Preprocess image
        image_rgb, gray_image, binary_mask = self.preprocess_image(image_path)

        # Extract features
        color_features = self.extract_color_features(image_rgb)
        texture_glcm = self.extract_texture_features_glcm(gray_image, binary_mask)
        texture_lbp = self.extract_texture_features_lbp(gray_image, binary_mask)
        size_shape_features = self.extract_size_shape_features(binary_mask)

        # Combine all features
        all_features = {
            **color_features,
            **texture_glcm,
            **texture_lbp,
            **size_shape_features
        }

        return all_features

    def extract_features_to_dataframe(self, image_paths, rice_names=None):
        """
        Extract features from multiple images and return as DataFrame.

        Parameters:
        -----------
        image_paths : list
            List of image file paths
        rice_names : list, optional
            List of rice variety names corresponding to each image

        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted features
        """
        all_features_list = []

        for idx, image_path in enumerate(image_paths):
            try:
                features = self.extract_all_features(image_path)
                if rice_names and idx < len(rice_names):
                    features['Name'] = rice_names[idx]
                all_features_list.append(features)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

        df = pd.DataFrame(all_features_list)
        return df


# Indian Rice Varieties (exact names as specified)
INDIAN_RICE_VARIETIES = [
    'Basmati rice',
    'Colam rice',  # Note: lowercase 'c' as specified
    'Indrayani rice',
    'Joha rice',
    'Matta rice',
    'Sona Masuri rice',
    'Kala Jira rice',
    'Ambemohar rice',
    'Ponni rice',
    'Jasmine rice',
    'Bamboo rice',
    'Mogara rice',
    'Brown rice'
]


def get_rice_variety_mapping():
    """
    Get mapping between rice variety names and class indices.

    Returns:
    --------
    dict
        Dictionary mapping variety names to class indices
    """
    return {variety: idx for idx, variety in enumerate(INDIAN_RICE_VARIETIES)}


if __name__ == "__main__":
    # Example usage
    extractor = RiceImageFeatureExtractor()
    print("Rice Image Feature Extractor initialized")
    print(f"Total features: {len(extractor.feature_names)}")
    print(f"Indian Rice Varieties: {len(INDIAN_RICE_VARIETIES)}")
