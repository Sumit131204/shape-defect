import React, { useState } from "react";
import { Card, Table } from "react-bootstrap";

const SizeResult = ({ measurements }) => {
  const [isHovered, setIsHovered] = useState(false);

  // Make sure we have valid measurements to work with
  const validMeasurements = Array.isArray(measurements) ? measurements : [];

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
            backgroundColor: "var(--light-blue)",
            color: "var(--primary-black)",
            borderBottom: "1px solid rgba(0,0,0,0.05)",
            position: "relative",
            zIndex: 1,
            transform: isHovered ? "translateZ(10px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
          }}
        >
          <h4 className="mb-0">Main Shape Size</h4>
        </Card.Header>
        <Card.Body
          style={{
            position: "relative",
            zIndex: 0,
            transform: isHovered ? "translateZ(5px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
          }}
        >
          {validMeasurements.length === 0 ? (
            <p className="text-muted">No measurements available.</p>
          ) : (
            <div>
              <p>Measurements of the main shape in the image:</p>
              <Table
                striped
                bordered
                hover
                responsive
                style={{
                  transform: isHovered ? "translateZ(8px)" : "translateZ(0)",
                  transition: "transform 0.3s ease",
                }}
              >
                <thead>
                  <tr
                    style={{
                      backgroundColor: "var(--light-blue)",
                      color: "var(--primary-black)",
                    }}
                  >
                    <th>#</th>
                    <th>Shape</th>
                    <th>Width</th>
                    <th>Height</th>
                    <th>Area</th>
                  </tr>
                </thead>
                <tbody>
                  {validMeasurements.map((item, index) => {
                    // Safe access to properties with fallbacks
                    const shape = item?.shape || "Unknown";

                    // Support both new (cm) and legacy (mm) formats
                    const diameterCm = item?.diameter_cm;
                    const widthCm = item?.width_cm;
                    const heightCm = item?.height_cm;
                    const areaCm2 = item?.area_cm2;

                    // Check for legacy format (mm)
                    const lengthMm = item?.length_mm;
                    const breadthMm = item?.breadth_mm;
                    const areaMm2 = item?.area_mm2;

                    // Determine if we're using cm or mm
                    const usingCentimeters =
                      diameterCm !== undefined ||
                      widthCm !== undefined ||
                      heightCm !== undefined ||
                      areaCm2 !== undefined;

                    // Format values and add units
                    const formatValue = (val, unit) => {
                      return val !== undefined && val !== null
                        ? `${val.toFixed(2)} ${unit}`
                        : "N/A";
                    };

                    // Get the appropriate measurements based on format
                    const width = usingCentimeters
                      ? diameterCm !== undefined
                        ? diameterCm
                        : widthCm
                      : lengthMm !== undefined
                      ? lengthMm / 10
                      : undefined; // Convert mm to cm if needed

                    const height = usingCentimeters
                      ? diameterCm !== undefined
                        ? null
                        : heightCm
                      : breadthMm !== undefined
                      ? breadthMm / 10
                      : undefined; // Convert mm to cm if needed

                    const area = usingCentimeters
                      ? areaCm2
                      : areaMm2 !== undefined
                      ? areaMm2 / 100
                      : undefined; // Convert mm² to cm² if needed

                    return (
                      <tr
                        key={index}
                        style={{
                          transform: isHovered
                            ? `translateZ(${5 - index * 0.5}px)`
                            : "translateZ(0)",
                          transition: "transform 0.3s ease",
                        }}
                      >
                        <td>{index + 1}</td>
                        <td>{shape}</td>
                        <td>
                          {diameterCm !== undefined && diameterCm !== null
                            ? formatValue(diameterCm, "cm")
                            : widthCm !== undefined && widthCm !== null
                            ? formatValue(widthCm, "cm")
                            : lengthMm !== undefined && lengthMm !== null
                            ? formatValue(lengthMm / 10, "cm")
                            : "N/A"}
                        </td>
                        <td>
                          {diameterCm !== undefined && diameterCm !== null
                            ? "-"
                            : heightCm !== undefined && heightCm !== null
                            ? formatValue(heightCm, "cm")
                            : breadthMm !== undefined && breadthMm !== null
                            ? formatValue(breadthMm / 10, "cm")
                            : "N/A"}
                        </td>
                        <td>
                          {areaCm2 !== undefined && areaCm2 !== null
                            ? formatValue(areaCm2, "cm²")
                            : areaMm2 !== undefined && areaMm2 !== null
                            ? formatValue(areaMm2 / 100, "cm²")
                            : "N/A"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </Table>
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
};

export default SizeResult;
