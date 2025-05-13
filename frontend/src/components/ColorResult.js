import React, { useState } from "react";
import { Card, Table } from "react-bootstrap";

const ColorResult = ({ colors }) => {
  const [isHovered, setIsHovered] = useState(false);

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
          <h4 className="mb-0">Color Detection Results</h4>
        </Card.Header>
        <Card.Body
          style={{
            position: "relative",
            zIndex: 0,
            transform: isHovered ? "translateZ(5px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
          }}
        >
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
                <th>Shape</th>
                <th>Color</th>
                <th>Sample</th>
                <th>RGB Value</th>
              </tr>
            </thead>
            <tbody>
              {colors.map((color, index) => (
                <tr
                  key={index}
                  style={{
                    transform: isHovered
                      ? `translateZ(${5 - index * 0.5}px)`
                      : "translateZ(0)",
                    transition: "transform 0.3s ease",
                  }}
                >
                  <td>{color.shape}</td>
                  <td>{color.color}</td>
                  <td>
                    <div
                      style={{
                        backgroundColor: color.hex,
                        width: "30px",
                        height: "30px",
                        borderRadius: "4px",
                        border: "1px solid #ddd",
                        margin: "0 auto",
                        boxShadow: isHovered
                          ? "0 2px 4px rgba(0,0,0,0.1)"
                          : "none",
                        transition: "box-shadow 0.3s ease",
                      }}
                    />
                  </td>
                  <td>
                    <code>{`(${color.rgb[0]}, ${color.rgb[1]}, ${color.rgb[2]})`}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>

          {colors.length === 0 && (
            <div className="text-center text-muted">
              <p>No colors detected.</p>
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
};

export default ColorResult;
