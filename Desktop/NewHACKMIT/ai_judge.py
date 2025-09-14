import gradio as gr
import numpy as np
from PIL import Image, ImageFilter, ImageStat
import cv2
import os
from scipy import ndimage, stats
from skimage import feature, filters, measure, segmentation
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy.fft import fft2, fftshift
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AIImageJudge:
    def __init__(self):
        self.criteria = {
            'pixel_consistency': 0,
            'compression_artifacts': 0,
            'noise_patterns': 0,
            'edge_coherence': 0,
            'color_distribution': 0,
            'texture_analysis': 0,
            'symmetry_analysis': 0,
            'frequency_domain': 0,
            'statistical_anomalies': 0,
            'gan_artifacts': 0,
            'diffusion_patterns': 0,
            'upsampling_detection': 0
        }
        
    def analyze_pixel_consistency(self, image):
        """Analyze pixel-level consistency patterns typical in AI images"""
        img_array = np.array(image)
        
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
       
        kernel = np.ones((5,5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        variance_uniformity = np.std(local_variance)
        
        
        score = 0
        if laplacian_var < 150:  
            score += 40
        if laplacian_var < 50:  
            score += 45
        if laplacian_var < 10: 
            score += 60
        if variance_uniformity < 80:  
            score += 35
        if variance_uniformity < 30:  
            score += 50
        if variance_uniformity < 5:  
            score += 70
            
   
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_std = np.std(gradient_magnitude)
        gradient_mean = np.mean(gradient_magnitude)
        
        if gradient_std < 15:  
            score += 30
        if gradient_std < 5: 
            score += 50
        if gradient_mean > 80:  
            score += 25
            
      
        pixel_std = np.std(gray)
        if pixel_std > 90:  
            score += 20
        if pixel_std > 120:  
            score += 30
            
        return min(score, 100)
    
    def analyze_compression_artifacts(self, image):
        """Detect JPEG compression artifacts - real photos usually have more"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        
        h, w = gray.shape
        block_artifacts = 0
        very_clean_blocks = 0
        
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
               
                high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
                total_energy = np.sum(np.abs(dct_block))
                
                if total_energy > 0:
                    ratio = high_freq_energy / total_energy
                    if ratio < 0.15: 
                        block_artifacts += 1
                    if ratio < 0.05:  
                        very_clean_blocks += 1
        
        total_blocks = ((h//8) * (w//8))
        if total_blocks > 0:
            artifact_ratio = block_artifacts / total_blocks
            clean_ratio = very_clean_blocks / total_blocks
            score = artifact_ratio * 80 + clean_ratio * 60
            return min(score, 100)
        
        return 0
    
    def analyze_noise_patterns(self, image):
        """Analyze noise patterns - AI images often lack natural sensor noise"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
       
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        noise = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
       
        noise_std = np.std(noise)
        noise_mean = np.mean(np.abs(noise))
        
       
        score = 0
        if noise_std < 8:  
            score += 45
        if noise_std < 3:  
            score += 55
        if noise_std < 0.5:  
            score += 75
        if noise_mean < 4:  
            score += 40
        if noise_mean < 1:  
            score += 60
        if noise_mean < 0.2: 
            score += 80
            
       
        noise_hist, _ = np.histogram(noise.flatten(), bins=50)
        noise_entropy = -np.sum((noise_hist + 1e-10) * np.log2(noise_hist + 1e-10))
        if noise_entropy < 4:  # Reduced threshold
            score += 25
        if noise_entropy < 2:  # Very organized = AI
            score += 45
            
        # Additional noise pattern analysis
        noise_skewness = stats.skew(noise.flatten())
        noise_kurtosis = stats.kurtosis(noise.flatten())
        
        if abs(noise_skewness) < 0.05:  # Very symmetric = AI
            score += 20
        if abs(noise_kurtosis) < 0.5:  # Too normal distribution = AI
            score += 15
            
        return min(score, 100)
    
    def analyze_edge_coherence(self, image):
        """Analyze edge patterns - AI images sometimes have inconsistent edges"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Analyze edge connectivity and coherence
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 60  # Suspicious lack of edges
            
        # Check for unnaturally smooth contours
        smooth_contours = 0
        for contour in contours:
            if len(contour) > 10:
                # Calculate contour smoothness
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                if area > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.8:  # Too circular/smooth
                        smooth_contours += 1
        
        if len(contours) > 0:
            smooth_ratio = smooth_contours / len(contours)
            return min(smooth_ratio * 80, 100)
        
        return 0
    
    def analyze_color_distribution(self, image):
        """Analyze color distribution patterns"""
        img_array = np.array(image)
        
        if len(img_array.shape) != 3:
            return 0
            
        # Analyze color histogram
        hist_r = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_array], [2], None, [256], [0, 256])
        
        # Check for unnatural color distributions
        score = 0
        
        # AI images often have too-perfect color gradients
        for hist in [hist_r, hist_g, hist_b]:
            hist_smooth = np.convolve(hist.flatten(), np.ones(7)/7, mode='same')
            smoothness = np.std(hist - hist_smooth.reshape(-1, 1))
            if smoothness < 20:  # Increased threshold
                score += 25
            if smoothness < 5:  # Very smooth = AI
                score += 30
        
        # Check for AI's characteristic color saturation patterns
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        sat_std = np.std(saturation)
        
        if sat_std > 45:  # AI often has high saturation variance
            score += 20
        if sat_std > 60:  # Very high = likely AI
            score += 25
            
        # Check for unnatural color clustering
        colors_reshaped = img_array.reshape(-1, 3)
        unique_colors = len(np.unique(colors_reshaped.view(np.dtype((np.void, colors_reshaped.dtype.itemsize*3)))))
        total_pixels = colors_reshaped.shape[0]
        color_diversity = unique_colors / total_pixels
        
        if color_diversity > 0.8:  # Higher threshold - real photos can have many colors
            score += 30
        if color_diversity > 0.95:  # Extremely high diversity = AI
            score += 50
            
        # Check for AI's characteristic color banding
        for channel in [img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]]:
            channel_diff = np.diff(channel.flatten())
            zero_diffs = np.sum(channel_diff == 0)
            total_diffs = len(channel_diff)
            if total_diffs > 0:
                banding_ratio = zero_diffs / total_diffs
                if banding_ratio > 0.3:  # Too much banding = AI
                    score += 20
                if banding_ratio > 0.5:  # Extreme banding = AI
                    score += 40
        
        return min(score, 100)
    
    def analyze_texture_patterns(self, image):
        """Analyze texture patterns using Local Binary Patterns"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Calculate Local Binary Pattern
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Analyze LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # AI images often have less texture variety
        entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
        max_entropy = np.log2(len(lbp_hist))
        
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
            if normalized_entropy < 0.6:  # Too little texture variety
                return min((0.6 - normalized_entropy) * 150, 100)
        
        return 0
    
    def analyze_symmetry(self, image):
        """Analyze unnatural symmetry patterns"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        h, w = gray.shape
        
        # Check horizontal symmetry
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        if left_half.size > 0 and right_half.size > 0:
            h_symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            if np.isnan(h_symmetry):
                h_symmetry = 0
        else:
            h_symmetry = 0
            
        # Check vertical symmetry
        top_half = gray[:h//2, :]
        bottom_half = np.flipud(gray[h//2:, :])
        
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        if top_half.size > 0 and bottom_half.size > 0:
            v_symmetry = np.corrcoef(top_half.flatten(), bottom_half.flatten())[0, 1]
            if np.isnan(v_symmetry):
                v_symmetry = 0
        else:
            v_symmetry = 0
        
        # High symmetry can indicate AI generation
        max_symmetry = max(abs(h_symmetry), abs(v_symmetry))
        if max_symmetry > 0.7:
            return min((max_symmetry - 0.7) * 200, 100)
        
        return 0
    
    def analyze_frequency_domain(self, image):
        """Analyze frequency domain characteristics"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Analyze frequency distribution
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Create rings to analyze frequency content
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Analyze high vs low frequency content
        low_freq_mask = distances < min(h, w) * 0.1
        high_freq_mask = distances > min(h, w) * 0.3
        
        low_freq_energy = np.mean(magnitude_spectrum[low_freq_mask])
        high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
        
        score = 0
        if low_freq_energy > 0:
            freq_ratio = high_freq_energy / low_freq_energy
            # AI images often have unusual frequency distributions
            if freq_ratio < 0.3:  # Too little high frequency content
                score = (0.3 - freq_ratio) * 200
        
        return min(score, 100)
    
    def analyze_real_photo_indicators(self, image):
        """Look for indicators that suggest a real photograph"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        real_score = 0
        
        # Check for natural sensor noise patterns
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        noise = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        noise_randomness = np.std(noise) / (np.mean(np.abs(noise)) + 1e-10)
        
        if noise_randomness > 2:  # Natural randomness = real photo
            real_score += 30
        if noise_randomness > 3:  # High randomness = definitely real
            real_score += 20
            
        # Check for natural JPEG compression artifacts
        h, w = gray.shape
        block_variance = []
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = gray[i:i+8, j:j+8]
                block_variance.append(np.var(block))
        
        if len(block_variance) > 0:
            variance_std = np.std(block_variance)
            if variance_std > 100:  # Natural variance = real photo
                real_score += 25
                
        # Check for natural edge imperfections
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Real photos have more irregular contours
            irregular_contours = 0
            for contour in contours:
                if len(contour) > 10:
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    if area > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity < 0.3:  # Irregular = real
                            irregular_contours += 1
            
            if len(contours) > 0:
                irregularity_ratio = irregular_contours / len(contours)
                if irregularity_ratio > 0.6:  # Many irregular contours = real
                    real_score += 20
                    
        return min(real_score, 100)
    
    def analyze_statistical_anomalies(self, image):
        """Detect statistical patterns typical of AI generation"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        score = 0
        
        # Check for unnatural statistical distributions
        pixel_values = gray.flatten()
        
        # Kolmogorov-Smirnov test against uniform distribution
        ks_stat, _ = stats.kstest(pixel_values, 'uniform')
        if ks_stat < 0.3:  # Too close to uniform = AI
            score += 40
            
        # Check for AI's characteristic value clustering
        hist, bins = np.histogram(pixel_values, bins=256)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, distance=10)
        peak_count = len(peaks)
        if peak_count > 20:  # Too many peaks = AI
            score += 35
            
        # Benford's law violation (AI often violates natural number distributions)
        first_digits = [int(str(int(p))[0]) for p in pixel_values if p > 0 and str(int(p))[0] != '0']
        if len(first_digits) > 100:
            digit_counts = np.bincount(first_digits)[1:]
            expected_benford = [np.log10(1 + 1/d) for d in range(1, 10)]
            if len(digit_counts) >= 9:
                digit_freq = digit_counts / np.sum(digit_counts)
                benford_deviation = np.sum(np.abs(digit_freq - expected_benford))
                if benford_deviation > 0.5:  # Violates Benford's law = AI
                    score += 30
                    
        return min(score, 100)
    
    def analyze_gan_artifacts(self, image):
        """Detect GAN-specific artifacts"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        score = 0
        
        # Check for checkerboard artifacts (common in GANs)
        kernel_checkerboard = np.array([[1, -1], [-1, 1]])
        checkerboard_response = cv2.filter2D(gray.astype(np.float32), -1, kernel_checkerboard)
        checkerboard_energy = np.mean(np.abs(checkerboard_response))
        
        if checkerboard_energy > 5:  # Checkerboard artifacts = GAN
            score += 50
        if checkerboard_energy > 10:  # Strong artifacts = definitely GAN
            score += 40
            
        # Check for mode collapse patterns
        # Divide image into patches and check for repetitive patterns
        h, w = gray.shape
        patch_size = 32
        patches = []
        
        for i in range(0, h-patch_size, patch_size//2):
            for j in range(0, w-patch_size, patch_size//2):
                patch = gray[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
                
        if len(patches) > 10:
            patches = np.array(patches)
            # Check for too similar patches (mode collapse)
            correlations = np.corrcoef(patches)
            high_corr_count = np.sum(correlations > 0.9) - len(patches)  # Exclude diagonal
            if high_corr_count > len(patches) * 0.3:  # Too many similar patches = GAN
                score += 45
                
        return min(score, 100)
    
    def analyze_diffusion_patterns(self, image):
        """Detect diffusion model artifacts"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        score = 0
        
        # Diffusion models often have characteristic noise residuals
        # Apply Gaussian blur and check residuals
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0)
        residual = gray.astype(np.float32) - blurred
        
        residual_std = np.std(residual)
        residual_mean = np.mean(np.abs(residual))
        
        # Diffusion models have specific residual patterns
        if residual_std < 3:  # Too clean residuals = diffusion
            score += 55
        if residual_mean < 1.5:  # Very clean = diffusion
            score += 45
            
        # Check for diffusion's characteristic frequency patterns
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Diffusion models often have specific frequency signatures
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Create frequency rings
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Check mid-frequency energy (diffusion models have characteristic patterns)
        mid_freq_mask = (distances > min(h, w) * 0.1) & (distances < min(h, w) * 0.4)
        mid_freq_energy = np.mean(magnitude_spectrum[mid_freq_mask])
        total_energy = np.mean(magnitude_spectrum)
        
        if total_energy > 0:
            mid_freq_ratio = mid_freq_energy / total_energy
            if mid_freq_ratio > 1.2:  # Unusual mid-frequency energy = diffusion
                score += 40
                
        return min(score, 100)
    
    def analyze_upsampling_detection(self, image):
        """Detect AI upsampling artifacts"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        score = 0
        
        # Check for interpolation artifacts
        # Downsample and upsample, then compare
        h, w = gray.shape
        downsampled = cv2.resize(gray, (w//2, h//2), interpolation=cv2.INTER_AREA)
        upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Calculate similarity
        diff = np.abs(gray.astype(np.float32) - upsampled.astype(np.float32))
        similarity = 1 - (np.mean(diff) / 255)
        
        if similarity > 0.95:  # Too similar to upsampled version = AI upscaling
            score += 60
        if similarity > 0.98:  # Extremely similar = definitely AI
            score += 40
            
        # Check for regular patterns (AI upsampling artifacts)
        # Use autocorrelation to detect repetitive patterns
        autocorr = cv2.matchTemplate(gray, gray[::2, ::2], cv2.TM_CCOEFF_NORMED)
        max_autocorr = np.max(autocorr)
        
        if max_autocorr > 0.99:  # Perfect repetition = AI
            score += 50
            
        return min(score, 100)
    
    def judge_image(self, image):
        """Main judging function that analyzes all criteria"""
        if image is None:
            return "No image provided", {}
            
        # Run all analysis functions
        self.criteria['pixel_consistency'] = self.analyze_pixel_consistency(image)
        self.criteria['compression_artifacts'] = self.analyze_compression_artifacts(image)
        self.criteria['noise_patterns'] = self.analyze_noise_patterns(image)
        self.criteria['edge_coherence'] = self.analyze_edge_coherence(image)
        self.criteria['color_distribution'] = self.analyze_color_distribution(image)
        self.criteria['texture_analysis'] = self.analyze_texture_patterns(image)
        self.criteria['symmetry_analysis'] = self.analyze_symmetry(image)
        self.criteria['frequency_domain'] = self.analyze_frequency_domain(image)
        self.criteria['statistical_anomalies'] = self.analyze_statistical_anomalies(image)
        self.criteria['gan_artifacts'] = self.analyze_gan_artifacts(image)
        self.criteria['diffusion_patterns'] = self.analyze_diffusion_patterns(image)
        self.criteria['upsampling_detection'] = self.analyze_upsampling_detection(image)
        
        # Check for real photo indicators to reduce false positives
        real_photo_score = self.analyze_real_photo_indicators(image)
        
        # EXTREME weighting - heavily favor AI detection
        weights = {
            'pixel_consistency': 0.20,
            'noise_patterns': 0.18,
            'compression_artifacts': 0.15,
            'statistical_anomalies': 0.12,
            'gan_artifacts': 0.10,
            'diffusion_patterns': 0.10,
            'upsampling_detection': 0.08,
            'edge_coherence': 0.04,
            'color_distribution': 0.02,
            'texture_analysis': 0.01,
            'symmetry_analysis': 0.00,
            'frequency_domain': 0.00
        }
        
        total_score = sum(self.criteria[key] * weights[key] for key in weights)
        
        # Apply real photo correction
        if real_photo_score > 40:  # Strong real photo indicators
            total_score -= 20
        elif real_photo_score > 20:  # Some real photo indicators
            total_score -= 10
            
        # Balanced bonus scoring
        high_scores = sum(1 for score in self.criteria.values() if score > 40)  # Higher threshold
        medium_scores = sum(1 for score in self.criteria.values() if score > 25)
        any_high = sum(1 for score in self.criteria.values() if score > 60)
        
        if high_scores >= 3:  # Need more indicators
            total_score += 15
        if high_scores >= 5:
            total_score += 25
        if medium_scores >= 7:  # Need more medium scores
            total_score += 10
        if any_high >= 1:
            total_score += 20  # Reduced bonus
            
        # Balanced thresholds - reduce false positives
        if total_score > 45:  # Higher threshold for high confidence
            verdict = "🤖 AI GENERATED"
            confidence = "HIGH"
        elif total_score > 25:  # Moderate threshold
            verdict = "🤖 LIKELY AI GENERATED"
            confidence = "MEDIUM"
        elif total_score > 15:  # Lower threshold for uncertainty
            verdict = "❓ UNCERTAIN"
            confidence = "LOW"
        else:
            verdict = "📷 LIKELY REAL"
            confidence = "MEDIUM"
            
        # Create detailed analysis report
        analysis_details = f"""
## 🔍 ANALYSIS REPORT
**VERDICT: {verdict}**
**CONFIDENCE: {confidence}**
**OVERALL SCORE: {total_score:.1f}/100**

### 📊 Detailed Criteria Analysis:
"""
        
        for criterion, score in self.criteria.items():
            status = "🔴 SUSPICIOUS" if score > 40 else "🟡 MODERATE" if score > 20 else "🟢 NORMAL"
            criterion_name = criterion.replace('_', ' ').title()
            analysis_details += f"- **{criterion_name}**: {score:.1f}/100 {status}\n"
        
        analysis_details += f"""
### 🧠 Analysis Methodology:
- **Pixel Consistency**: Checks for unnatural smoothness and variance patterns
- **Compression Artifacts**: Analyzes JPEG compression patterns typical in real photos
- **Noise Patterns**: Detects natural sensor noise vs artificial cleanliness
- **Edge Coherence**: Examines edge patterns and contour characteristics
- **Color Distribution**: Analyzes color histogram naturalness
- **Texture Analysis**: Uses Local Binary Patterns to assess texture variety
- **Symmetry Analysis**: Detects unnatural symmetrical patterns
- **Frequency Domain**: Analyzes frequency distribution characteristics

### 📝 Notes:
- Scores above 45 indicate strong AI generation likelihood (BALANCED DETECTION)
- Multiple high scores across criteria increase confidence
- System now includes real photo validation to reduce false positives
- Calibrated based on user feedback to balance AI detection with real photo accuracy
- Uses 12 AI detection algorithms plus real photo validation patterns
"""
        
        return analysis_details, self.criteria

def create_judge_interface():
    judge = AIImageJudge()
    
    def analyze_images(img1, img2, img3, img4, img5):
        images = [img1, img2, img3, img4, img5]
        results = []
        
        for i, img in enumerate(images, 1):
            if img is not None:
                analysis, criteria = judge.judge_image(img)
                results.append(f"## 🖼️ IMAGE {i}\n{analysis}\n---\n")
            else:
                results.append(f"## 🖼️ IMAGE {i}\n*No image provided*\n---\n")
        
        return "\n".join(results)
    
    # Create Gradio interface
    with gr.Blocks(title="AI Image Judge", theme=gr.themes.Monochrome()) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🤖 AI IMAGE JUDGE</h1>
            <p style="font-size: 18px;">Advanced AI Detection System</p>
            <p>Upload up to 5 images for comprehensive AI generation analysis</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>📤 Upload Images for Analysis</h3>")
                img1 = gr.Image(label="Image 1", type="pil")
                img2 = gr.Image(label="Image 2", type="pil")
                img3 = gr.Image(label="Image 3", type="pil")
                img4 = gr.Image(label="Image 4", type="pil")
                img5 = gr.Image(label="Image 5", type="pil")
                
                analyze_btn = gr.Button("🔍 ANALYZE IMAGES", variant="primary", size="lg")
        
        with gr.Column():
            gr.HTML("<h3>📋 Analysis Results</h3>")
            results = gr.Markdown(
                value="Upload images and click 'ANALYZE IMAGES' to begin the analysis...",
                label="Detailed Analysis Report"
            )
        
        analyze_btn.click(
            analyze_images,
            inputs=[img1, img2, img3, img4, img5],
            outputs=results
        )
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #ddd;">
            <p><strong>🔬 Analysis Criteria:</strong></p>
            <p>Pixel Consistency • Compression Artifacts • Noise Patterns • Edge Coherence</p>
            <p>Color Distribution • Texture Analysis • Symmetry Analysis • Frequency Domain</p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_judge_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True
    )
