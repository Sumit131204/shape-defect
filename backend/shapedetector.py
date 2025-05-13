# import the necessary packages
import cv2
import numpy as np

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
		
		# Calculate shape metrics for better classification
		# Circularity: 4πArea/Perimeter²
		area = cv2.contourArea(c)
		circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
		
		# Solidity: ratio of contour area to its convex hull area
		hull = cv2.convexHull(c)
		hull_area = cv2.contourArea(hull)
		solidity = area / hull_area if hull_area > 0 else 0
		
		# Aspect ratio of the bounding box
		(x, y, w, h) = cv2.boundingRect(approx)
		ar = w / float(h) if h > 0 else 0

		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"

		# if the shape has 4 vertices, it is either a square or a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"
   
		# if the shape is a hexagon, it will have 6 vertices
		elif len(approx) == 6:
			shape = "hexagon"

		# if it's a circle, it will have many vertices and high circularity
		elif circularity > 0.85 and solidity > 0.85:
			shape = "circle"
			
		# otherwise, we assume it's an irregular shape
		else:
			shape = "irregular"

		# return the name of the shape
		return shape