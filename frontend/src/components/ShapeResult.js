import React, { useState } from "react";
import { Card, ListGroup, Badge } from "react-bootstrap";

const ShapeResult = ({ shapes }) => {
  const [isHovered, setIsHovered] = useState(false);

  // Function to get badge color based on shape type
  const getBadgeColor = (shape) => {
    const colorMap = {
      triangle: "warning",
      square: "primary",
      rectangle: "info",
      pentagon: "success",
      circle: "danger",
      ellipse: "secondary",
      hexagon: "dark",
      irregular: "light",
    };
    return colorMap[shape] || "secondary";
  };

  return (
    <div
      style={{
        perspective: "1000px",
        marginBottom: "20px",
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Card
        style={{
          transform: isHovered
            ? "rotateY(-5deg) rotateX(5deg)"
            : "rotateY(0deg) rotateX(0deg)",
          transition: "transform 0.3s ease, box-shadow 0.3s ease",
          transformStyle: "preserve-3d",
          boxShadow: isHovered
            ? "rgba(0, 0, 0, 0.1) 5px 5px 15px, rgba(0, 0, 0, 0.07) 15px 15px 20px"
            : "rgba(0, 0, 0, 0.1) 0px 4px 12px, rgba(0, 0, 0, 0.05) 0px 1px 3px",
          borderRadius: "10px",
          border: "1px solid rgba(255,255,255,0.2)",
          overflow: "hidden",
        }}
      >
        <Card.Header
          style={{
            borderBottom: "1px solid rgba(0,0,0,0.05)",
            position: "relative",
            zIndex: 1,
            transform: isHovered ? "translateZ(10px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
            backgroundColor: "var(--light-blue)",
            color: "var(--primary-black)",
          }}
        >
          <h4 className="mb-0">Detected Shapes</h4>
        </Card.Header>
        <Card.Body
          style={{
            position: "relative",
            zIndex: 0,
            transform: isHovered ? "translateZ(5px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
          }}
        >
          {shapes.length === 0 ? (
            <p className="text-muted">No shapes detected in the image.</p>
          ) : (
            <div>
              <p>Found {shapes.length} shape(s) in the image:</p>
              <ListGroup variant="flush">
                {shapes.map((item, index) => (
                  <ListGroup.Item
                    key={index}
                    className="d-flex flex-column"
                    style={{
                      transform: isHovered
                        ? `translateZ(${15 - index * 2}px)`
                        : "translateZ(0)",
                      transition: "transform 0.3s ease",
                      borderRadius: "6px",
                      marginBottom: "8px",
                      boxShadow: isHovered
                        ? "0 2px 4px rgba(0,0,0,0.05)"
                        : "none",
                    }}
                  >
                    <div className="d-flex justify-content-between align-items-center">
                      <div>
                        <Badge
                          bg={getBadgeColor(item.shape)}
                          className="me-2"
                          style={{
                            transform: isHovered
                              ? "translateZ(5px) scale(1.05)"
                              : "translateZ(0) scale(1)",
                            transition: "transform 0.3s ease",
                            boxShadow: isHovered
                              ? "0 2px 4px rgba(0,0,0,0.1)"
                              : "none",
                          }}
                        >
                          {item.shape.charAt(0).toUpperCase() +
                            item.shape.slice(1)}
                        </Badge>
                      </div>
                      {item.area_mm2 && (
                        <div
                          className="text-muted"
                          style={{
                            transform: isHovered
                              ? "translateZ(5px)"
                              : "translateZ(0)",
                            transition: "transform 0.3s ease",
                          }}
                        >
                          Area: {item.area_mm2.toFixed(1)} mm²
                        </div>
                      )}
                    </div>
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
};

export default ShapeResult;
