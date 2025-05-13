import React, { useState, useEffect } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Spinner,
  Alert,
  Toast,
  ToastContainer,
} from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";
import ImageUploader from "./components/ImageUploader";
import ImageDisplay from "./components/ImageDisplay";
import ShapeResult from "./components/ShapeResult";
import SizeResult from "./components/SizeResult";
import ColorResult from "./components/ColorResult";
import Settings from "./components/Settings";
import axios from "axios";

// Configure axios with default settings
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";
axios.defaults.baseURL = API_BASE_URL;

function App() {
  const [file, setFile] = useState(null);
  const [filename, setFilename] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [processedImageUrl, setProcessedImageUrl] = useState("");
  const [shapes, setShapes] = useState([]);
  const [measurements, setMeasurements] = useState([]);
  const [colors, setColors] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("upload"); // 'upload', 'settings', 'shape', 'size', 'color'
  const [debug, setDebug] = useState(null);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState("");

  // Utility function to fix image URLs from the backend
  const fixImageUrl = (url) => {
    if (!url) return "";

    setDebug((prev) => ({ ...prev, originalUrl: url }));

    // If it's a static URL path from our backend
    if (url.startsWith("/static")) {
      // In development, use API base URL
      const fixedUrl = `${API_BASE_URL}${url}`;
      setDebug((prev) => ({ ...prev, fixedUrl }));
      return fixedUrl;
    }

    // If it starts with http://, return as is
    if (url.startsWith("http://") || url.startsWith("https://")) {
      return url;
    }

    // Otherwise prepend the API base URL
    const fixedUrl = `${API_BASE_URL}/${
      url.startsWith("/") ? url.substring(1) : url
    }`;
    setDebug((prev) => ({ ...prev, fixedUrl }));
    return fixedUrl;
  };

  // Check if processedImageUrl is valid - perform error detection
  useEffect(() => {
    if (processedImageUrl) {
      const img = new Image();
      img.onload = () => {
        setDebug((prev) => ({ ...prev, imageLoaded: true }));
      };
      img.onerror = (e) => {
        setDebug((prev) => ({ ...prev, imageError: e.toString() }));
      };
      img.src = processedImageUrl;
    }
  }, [processedImageUrl]);

  const handleFileUpload = async (file) => {
    setFile(file);
    setProcessedImageUrl("");
    setShapes([]);
    setMeasurements([]);
    setColors([]);
    setError("");
    setDebug({});

    const fileUrl = URL.createObjectURL(file);
    setImageUrl(fileUrl);

    try {
      setLoading(true);
      setActiveTab("upload");

      const formData = new FormData();
      formData.append("image", file);

      const response = await axios.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setFilename(response.data.filename);
      setLoading(false);
    } catch (err) {
      setError("Error uploading image. Please try again.");
      setLoading(false);
    }
  };

  const handleDetectShape = async () => {
    if (!filename) {
      setError("Please upload an image first");
      return;
    }

    try {
      setLoading(true);
      setActiveTab("shape");
      setProcessedImageUrl("");
      setShapes([]);
      setError("");
      setDebug({});

      const response = await axios.post("/detect-shape", { filename });
      console.log("Shape detection response:", response.data);
      setDebug((prev) => ({ ...prev, shapeResponse: response.data }));

      // Handle different response formats
      if (response.data.shapes && response.data.shapes.length > 0) {
        // Original format with shapes array
        setShapes(response.data.shapes);
      } else if (response.data.result) {
        // Format with single result object
        if (typeof response.data.result === "object") {
          setShapes([response.data.result]);
        } else if (Array.isArray(response.data.result)) {
          setShapes(response.data.result);
        }
      }

      // Handle processed image URL with our utility function
      let processedUrl = null;
      if (response.data.processedImage) {
        processedUrl = response.data.processedImage;
      } else if (response.data.result && response.data.result.processed_image) {
        processedUrl = response.data.result.processed_image;
      } else if (response.data.image_url) {
        processedUrl = response.data.image_url;
      } else if (response.data.viz_path) {
        processedUrl = response.data.viz_path;
      }

      if (processedUrl) {
        const fixedUrl = fixImageUrl(processedUrl);
        setProcessedImageUrl(fixedUrl);
        setDebug((prev) => ({ ...prev, processedUrl, fixedUrl }));
      } else {
        setError("No processed image URL in the response");
        setDebug((prev) => ({
          ...prev,
          error: "No processed image URL found",
        }));
      }

      setLoading(false);

      // Check if shapes were detected
      if (
        (!response.data.shapes || response.data.shapes.length === 0) &&
        (!response.data.result ||
          (typeof response.data.result === "object" &&
            Object.keys(response.data.result).length === 0) ||
          (Array.isArray(response.data.result) &&
            response.data.result.length === 0))
      ) {
        setError(
          "No shapes were detected in the image. Please try another image."
        );
      }
    } catch (err) {
      console.error("Shape detection error:", err);
      // Get detailed error message from response if available
      const errorMessage =
        err.response && err.response.data && err.response.data.error
          ? err.response.data.error
          : "Error detecting shapes. Please try again.";

      setError(errorMessage);
      setDebug((prev) => ({
        ...prev,
        error: err.toString(),
        errorResponse: err.response ? err.response.data : null,
      }));
      setLoading(false);
    }
  };

  const handleDetectSize = async () => {
    if (!filename) {
      setError("Please upload an image first");
      return;
    }

    try {
      setLoading(true);
      setActiveTab("size");
      setProcessedImageUrl("");
      setMeasurements([]);
      setError("");
      setDebug({});

      const response = await axios.post("/detect-size", { filename });
      console.log("Size detection response:", response.data);
      setDebug((prev) => ({ ...prev, sizeResponse: response.data }));

      if (response.data.measurements) {
        setMeasurements(response.data.measurements);
      }

      // Handle processed image URL
      let processedUrl = null;
      if (response.data.processedImage) {
        processedUrl = response.data.processedImage;
      } else if (response.data.image_url) {
        processedUrl = response.data.image_url;
      } else if (response.data.size_image) {
        processedUrl = response.data.size_image;
      }

      if (processedUrl) {
        const fixedUrl = fixImageUrl(processedUrl);
        setProcessedImageUrl(fixedUrl);
        setDebug((prev) => ({ ...prev, processedUrl, fixedUrl }));
      } else {
        setError("No processed image URL in the response");
        setDebug((prev) => ({
          ...prev,
          error: "No processed image URL found",
        }));
      }

      setLoading(false);
    } catch (err) {
      console.error("Size detection error:", err);
      setError("Error detecting sizes. Please try again.");
      setDebug((prev) => ({ ...prev, error: err.toString() }));
      setLoading(false);
    }
  };

  const handleDetectColor = async () => {
    if (!filename) {
      setError("Please upload an image first");
      return;
    }

    try {
      setLoading(true);
      setActiveTab("color");
      setProcessedImageUrl("");
      setColors([]);
      setError("");
      setDebug({});

      const response = await axios.post("/detect-color", { filename });
      console.log("Color detection response:", response.data);
      setDebug((prev) => ({ ...prev, colorResponse: response.data }));

      if (response.data.colors) {
        setColors(response.data.colors);
      }

      // Handle processed image URL
      let processedUrl = null;
      if (response.data.processedImage) {
        processedUrl = response.data.processedImage;
      } else if (response.data.image_url) {
        processedUrl = response.data.image_url;
      } else if (response.data.color_image) {
        processedUrl = response.data.color_image;
      }

      if (processedUrl) {
        const fixedUrl = fixImageUrl(processedUrl);
        setProcessedImageUrl(fixedUrl);
        setDebug((prev) => ({ ...prev, processedUrl, fixedUrl }));
      } else {
        setError("No processed image URL in the response");
        setDebug((prev) => ({
          ...prev,
          error: "No processed image URL found",
        }));
      }

      setLoading(false);
    } catch (err) {
      console.error("Color detection error:", err);
      setError("Error detecting colors. Please try again.");
      setDebug((prev) => ({ ...prev, error: err.toString() }));
      setLoading(false);
    }
  };

  return (
    <Container
      fluid
      className="px-3 py-4"
      style={{ maxWidth: "1200px", margin: "0 auto" }}
    >
      <Card className="shadow-sm">
        <div className="logo-header">
          {/* <img
            src={`${process.env.PUBLIC_URL}/logo.jpg`}
            alt="Concept Systems Logo"
            className="company-logo"
          /> */}
          <span>
            Automated Realtime Image-Driven Quality Control:
            <br />
            For Industrial Object Classification
          </span>
        </div>
        <Card.Body className="px-4 py-4 position-relative">
          {/* Text image - positioned left in detection tabs, right in upload tab */}
          <div
            style={{
              position: "absolute",
              top: "70px",
              ...(activeTab === "upload" || activeTab === "settings"
                ? { right: "30px" }
                : { left: "30px" }),
              zIndex: "100",
            }}
          >
            <img
              src={`${process.env.PUBLIC_URL}/text.jpg`}
              alt="Text"
              style={{
                maxHeight:
                  activeTab === "upload" || activeTab === "settings"
                    ? "80px"
                    : "50px",
                borderRadius: "4px",
              }}
            />
          </div>

          <Row className="mb-4">
            <Col xs={12} className="d-flex justify-content-center">
              <div className="action-buttons d-flex flex-wrap justify-content-center">
                <Button
                  variant={
                    activeTab === "upload" ? "primary" : "outline-primary"
                  }
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={() => setActiveTab("upload")}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <path
                      d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Upload Image
                </Button>
                <Button
                  variant={
                    activeTab === "settings" ? "primary" : "outline-primary"
                  }
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={() => setActiveTab("settings")}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <path
                      d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z"
                      fill="currentColor"
                      strokeWidth="0"
                    />
                    <path
                      d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319z"
                      fill="currentColor"
                      strokeWidth="0"
                    />
                  </svg>
                  Settings
                </Button>
                <Button
                  variant={
                    activeTab === "shape" ? "primary" : "outline-primary"
                  }
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={handleDetectShape}
                  disabled={!filename}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <path
                      d="M3 3l18 18M10.5 10.5l3 3M5 13l6-6 1.243 1.243M10.984 10.984L16 6l2 2-5.016 5.016M5 19l5-5 5 5"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Detect Shape
                </Button>
                <Button
                  variant={activeTab === "size" ? "primary" : "outline-primary"}
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={handleDetectSize}
                  disabled={!filename}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <path
                      d="M21 6H3M21 12H3M21 18H3M8 6v12M16 6v12"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Detect Size
                </Button>
                <Button
                  variant={
                    activeTab === "color" ? "primary" : "outline-primary"
                  }
                  className="mx-2 my-1 d-flex align-items-center"
                  style={{ minWidth: "150px", borderRadius: "4px" }}
                  onClick={handleDetectColor}
                  disabled={!filename}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    className="me-2"
                  >
                    <circle cx="12" cy="12" r="10" strokeWidth="2" />
                    <path
                      d="M8 14s1.5 2 4 2 4-2 4-2"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <line
                      x1="9"
                      y1="9"
                      x2="9.01"
                      y2="9"
                      strokeWidth="3"
                      strokeLinecap="round"
                    />
                    <line
                      x1="15"
                      y1="9"
                      x2="15.01"
                      y2="9"
                      strokeWidth="3"
                      strokeLinecap="round"
                    />
                  </svg>
                  Detect Color
                </Button>
              </div>
            </Col>
          </Row>

          {error && (
            <Alert variant="danger" onClose={() => setError("")} dismissible>
              {error}
            </Alert>
          )}

          <Row className="justify-content-center">
            <Col lg={8} className="mb-4">
              {activeTab === "upload" && (
                <ImageUploader
                  onFileUpload={handleFileUpload}
                  imageUrl={imageUrl}
                />
              )}

              {activeTab === "settings" && <Settings />}

              {processedImageUrl && activeTab !== "upload" ? (
                <div className="mb-4">
                  <h5 className="mb-3 text-center">
                    {activeTab === "shape"
                      ? "Shape Detection Result"
                      : activeTab === "size"
                      ? "Main Shape Size Detection Result"
                      : "Color Detection Result"}
                  </h5>
                  <div className="text-center">
                    <img
                      src={processedImageUrl}
                      alt={`${activeTab} Detection Result`}
                      className="img-fluid rounded mb-3"
                      style={{
                        maxHeight: "400px",
                        border: "1px solid #dee2e6",
                        boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
                      }}
                    />
                    <div className="mt-2 mb-3">
                      <Button
                        variant="outline-primary"
                        className="download-btn"
                        style={{
                          transition: "all 0.3s ease",
                          borderRadius: "6px",
                          padding: "8px 16px",
                          boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                        }}
                        onMouseOver={(e) => {
                          e.currentTarget.style.transform = "translateY(-2px)";
                          e.currentTarget.style.boxShadow =
                            "0 4px 8px rgba(0,0,0,0.15)";
                        }}
                        onMouseOut={(e) => {
                          e.currentTarget.style.transform = "translateY(0)";
                          e.currentTarget.style.boxShadow =
                            "0 2px 4px rgba(0,0,0,0.1)";
                        }}
                        onClick={() => {
                          // Extract filename from the URL
                          const urlParts = processedImageUrl.split("/");
                          const filename = urlParts[urlParts.length - 1];

                          // Use the download endpoint
                          const downloadUrl = `${axios.defaults.baseURL}/download/${filename}`;

                          // Create a temporary anchor element
                          const link = document.createElement("a");
                          link.href = downloadUrl;
                          link.download = `${activeTab}-detection-result.png`;
                          document.body.appendChild(link);
                          link.click();
                          document.body.removeChild(link);

                          // Show toast notification
                          setToastMessage(
                            `${
                              activeTab.charAt(0).toUpperCase() +
                              activeTab.slice(1)
                            } detection result downloaded!`
                          );
                          setShowToast(true);
                        }}
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="16"
                          height="16"
                          fill="currentColor"
                          className="bi bi-download me-2"
                          viewBox="0 0 16 16"
                        >
                          <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z" />
                          <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z" />
                        </svg>
                        Download Image
                      </Button>
                    </div>
                    {debug && debug.imageError && (
                      <Alert variant="warning" className="mt-2 mb-0">
                        Error loading image: {debug.imageError}
                      </Alert>
                    )}
                  </div>
                </div>
              ) : null}

              {loading && (
                <div className="text-center py-5">
                  <Spinner animation="border" role="status" variant="primary">
                    <span className="visually-hidden">Loading...</span>
                  </Spinner>
                  <p className="mt-3">Processing image...</p>
                </div>
              )}
            </Col>

            {(activeTab === "shape" && shapes.length > 0) ||
            (activeTab === "size" && measurements.length > 0) ||
            (activeTab === "color" && colors.length > 0) ? (
              <Col lg={4}>
                {activeTab === "shape" && shapes.length > 0 && (
                  <ShapeResult shapes={shapes} />
                )}

                {activeTab === "size" && measurements.length > 0 && (
                  <SizeResult measurements={measurements} />
                )}

                {activeTab === "color" && colors.length > 0 && (
                  <ColorResult colors={colors} />
                )}
              </Col>
            ) : null}
          </Row>

          {process.env.NODE_ENV === "development" &&
            debug &&
            Object.keys(debug).length > 0 && (
              <Row className="mt-3">
                <Col>
                  <details>
                    <summary className="text-muted">Debug Information</summary>
                    <pre
                      className="bg-light p-3 mt-2 rounded"
                      style={{
                        fontSize: "12px",
                        maxHeight: "200px",
                        overflow: "auto",
                      }}
                    >
                      {JSON.stringify(debug, null, 2)}
                    </pre>
                  </details>
                </Col>
              </Row>
            )}
        </Card.Body>
        <Card.Footer className="text-center py-3">
          <p className="text-muted mb-0">
            Upload an image to detect its shape, size, and color.
          </p>
        </Card.Footer>
      </Card>

      <ToastContainer
        position="bottom-end"
        className="p-3"
        style={{ zIndex: 1050 }}
      >
        <Toast
          onClose={() => setShowToast(false)}
          show={showToast}
          delay={3000}
          autohide
          bg="success"
        >
          <Toast.Header>
            <strong className="me-auto">Success</strong>
          </Toast.Header>
          <Toast.Body className="text-white">{toastMessage}</Toast.Body>
        </Toast>
      </ToastContainer>
    </Container>
  );
}

export default App;
