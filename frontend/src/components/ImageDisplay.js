import React, { useState, useRef, useEffect } from "react";
import { Container, Row, Col, Card } from "react-bootstrap";

const ImageDisplay = ({
  originalImage,
  processedImage,
  shapeData,
  colorData,
  displayMode,
}) => {
  const [isHoveredOriginal, setIsHoveredOriginal] = useState(false);
  const [isHoveredProcessed, setIsHoveredProcessed] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const cardRef = useRef(null);

  // Handle mouse movement for dynamic 3D effect
  const handleMouseMove = (e) => {
    if (!cardRef.current) return;

    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Calculate percentage of mouse position within the card
    const xPercent = (x / rect.width - 0.5) * 2; // -1 to 1
    const yPercent = (y / rect.height - 0.5) * 2; // -1 to 1

    setMousePosition({ x: xPercent, y: yPercent });
  };

  // Reset mouse position when mouse leaves
  const handleMouseLeave = () => {
    setIsHoveredProcessed(false);
    setMousePosition({ x: 0, y: 0 });
  };

  // Generate a formatted string of shape data details
  const formatShapeDetails = () => {
    if (!shapeData || !shapeData.shapes) return "No shape data available";

    return shapeData.shapes.map((shape, index) => (
      <div key={index} className="mb-2">
        <strong>Shape {index + 1}:</strong> {shape.shape_type}{" "}
        {shape.dimensions && (
          <>
            <br />
            <strong>Size:</strong> {shape.dimensions.length?.toFixed(2)} mm ×{" "}
            {shape.dimensions.breadth?.toFixed(2)} mm
            <br />
            <strong>Area:</strong> {shape.dimensions.area?.toFixed(2)} mm²
          </>
        )}
      </div>
    ));
  };

  // Generate a formatted string of color data details
  const formatColorDetails = () => {
    if (!colorData || !colorData.colors) return "No color data available";

    return colorData.colors.map((color, index) => (
      <div key={index} className="mb-2">
        <strong>Color {index + 1}:</strong> {color.color_name}{" "}
        <div
          style={{
            display: "inline-block",
            width: "20px",
            height: "20px",
            backgroundColor: `rgb(${color.rgb[0]}, ${color.rgb[1]}, ${color.rgb[2]})`,
            border: "1px solid #ddd",
            marginLeft: "5px",
            verticalAlign: "middle",
          }}
        ></div>
        <br />
        <strong>RGB:</strong> {color.rgb.join(", ")}
        {color.shape_type && (
          <>
            <br />
            <strong>Shape:</strong> {color.shape_type}
          </>
        )}
      </div>
    ));
  };

  return (
    <Container>
      <Row className="my-4">
        {displayMode === "shape" && shapeData && (
          <Col md={6} className="mb-4">
            <Card className="h-100 shadow-sm">
              <Card.Body>
                <Card.Title>Shape Details</Card.Title>
                <div className="shape-details mt-3">{formatShapeDetails()}</div>
              </Card.Body>
            </Card>
          </Col>
        )}

        {displayMode === "color" && colorData && (
          <Col md={6} className="mb-4">
            <Card className="h-100 shadow-sm">
              <Card.Body>
                <Card.Title>Color Details</Card.Title>
                <div className="color-details mt-3">{formatColorDetails()}</div>
              </Card.Body>
            </Card>
          </Col>
        )}

        <Col md={displayMode === "none" ? 12 : 6} className="mb-4">
          <div
            ref={cardRef}
            className="processed-image-container"
            style={{
              perspective: "1000px",
              minHeight: "300px",
              cursor: "default",
            }}
            onMouseEnter={() => setIsHoveredProcessed(true)}
            onMouseLeave={handleMouseLeave}
            onMouseMove={handleMouseMove}
          >
            <h5 className="mb-3 text-center">
              {displayMode === "shape"
                ? "Detected Shapes"
                : displayMode === "color"
                ? "Detected Colors"
                : "Processed Image"}
            </h5>
            {processedImage ? (
              <div
                className="text-center"
                style={{
                  transform: isHoveredProcessed
                    ? `rotateY(${mousePosition.x * 10}deg) rotateX(${
                        -mousePosition.y * 10
                      }deg)`
                    : "rotateY(0deg) rotateX(0deg)",
                  transition: isHoveredProcessed
                    ? "transform 0.1s ease"
                    : "transform 0.5s ease-out",
                  transformStyle: "preserve-3d",
                  boxShadow: isHoveredProcessed
                    ? `rgba(0, 0, 0, 0.1) ${5 + mousePosition.x * 10}px ${
                        5 + mousePosition.y * 10
                      }px 15px, 
                       rgba(0, 0, 0, 0.07) ${15 + mousePosition.x * 15}px ${
                        15 + mousePosition.y * 15
                      }px 20px`
                    : "rgba(0, 0, 0, 0.1) 0px 4px 12px, rgba(0, 0, 0, 0.05) 0px 1px 3px",
                  borderRadius: "12px",
                  padding: "16px",
                  background: "#fff",
                  position: "relative",
                  borderTop: "1px solid rgba(255,255,255,0.5)",
                  borderLeft: "1px solid rgba(255,255,255,0.5)",
                  backdropFilter: "blur(5px)",
                  marginBottom: "10px",
                }}
              >
                <div
                  style={{
                    transform: isHoveredProcessed
                      ? `translateZ(50px)`
                      : "translateZ(0)",
                    transition: isHoveredProcessed
                      ? "transform 0.1s ease"
                      : "transform 0.5s ease-out",
                  }}
                >
                  <img
                    src={processedImage}
                    alt="Processed"
                    className="img-fluid rounded"
                    style={{
                      maxHeight: "300px",
                      objectFit: "contain",
                      boxShadow: isHoveredProcessed
                        ? "0 8px 16px rgba(0,0,0,0.15)"
                        : "0 2px 4px rgba(0,0,0,0.1)",
                      borderRadius: "8px",
                      transition: isHoveredProcessed
                        ? "box-shadow 0.1s ease"
                        : "box-shadow 0.5s ease-out",
                    }}
                  />
                </div>

                {isHoveredProcessed && (
                  <div
                    style={{
                      position: "absolute",
                      top: "50%",
                      left: "50%",
                      transform: "translate(-50%, -50%) translateZ(5px)",
                      width: "100%",
                      height: "100%",
                      background:
                        "radial-gradient(circle at center, transparent 60%, rgba(255,255,255,0.1) 100%)",
                      pointerEvents: "none",
                      opacity: 0.7,
                      borderRadius: "12px",
                    }}
                  />
                )}
              </div>
            ) : (
              <div className="text-center text-muted py-5">
                No processed image available
              </div>
            )}
          </div>
        </Col>
      </Row>
    </Container>
  );
};

export default ImageDisplay;
