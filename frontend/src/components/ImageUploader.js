import React, { useRef, useState } from "react";
import { Card, Button, Form } from "react-bootstrap";

const ImageUploader = ({ onFileUpload, imageUrl }) => {
  const [dragActive, setDragActive] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    inputRef.current.click();
  };

  return (
    <div className="d-flex flex-column align-items-center justify-content-center w-100 ">
      {/* Display the original image if available */}
      {imageUrl && (
        <div
          className="original-image-container mb-4"
          style={{
            maxWidth: "400px",
            perspective: "1000px",
          }}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <h5 className="mb-3 text-center">Original Image</h5>
          <div
            className="text-center"
            style={{
              transform: isHovered
                ? "rotateY(5deg) rotateX(5deg)"
                : "rotateY(0deg) rotateX(0deg)",
              transition: "transform 0.3s ease",
              transformStyle: "preserve-3d",
              boxShadow: isHovered
                ? "rgba(0, 0, 0, 0.1) -5px 5px 10px, rgba(0, 0, 0, 0.07) -15px 15px 20px"
                : "rgba(0, 0, 0, 0.1) 0px 4px 12px, rgba(0, 0, 0, 0.05) 0px 1px 3px",
              borderRadius: "8px",
              padding: "10px",
              background: "#fff",
              position: "relative",
              borderTop: "1px solid rgba(255,255,255,0.5)",
              borderLeft: "1px solid rgba(255,255,255,0.5)",
              backdropFilter: "blur(5px)",
            }}
          >
            <img
              src={imageUrl}
              alt="Original"
              className="img-fluid rounded"
              style={{
                maxHeight: "300px",
                maxWidth: "100%",
                objectFit: "contain",
                boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                transform: "translateZ(20px)",
                transition: "transform 0.3s ease",
              }}
            />
          </div>
        </div>
      )}

      {/* Upload area */}
      <div
        className={`upload-container ${dragActive ? "drag-active" : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        style={{
          border: dragActive
            ? "2px dashed var(--dark-blue)"
            : "2px dashed var(--light-blue)",
          borderRadius: "12px",
          padding: "40px 20px",
          textAlign: "center",
          cursor: "pointer",
          background: dragActive ? "rgba(142, 202, 230, 0.1)" : "#f8f9fa",
          transition: "all 0.3s ease",
          width: "100%",
          maxWidth: "500px",
          margin: "0 auto",
          perspective: "1000px",
          transformStyle: "preserve-3d",
          boxShadow: dragActive
            ? "rgba(0, 0, 0, 0.1) 0px 10px 15px -3px, rgba(0, 0, 0, 0.05) 0px 4px 6px -2px"
            : "rgba(0, 0, 0, 0.1) 0px 4px 12px, rgba(0, 0, 0, 0.05) 0px 1px 3px",
          transform: dragActive ? "translateY(-5px)" : "translateY(0)",
        }}
        onClick={onButtonClick}
      >
        <div
          className="upload-icon mb-4"
          style={{
            transform: dragActive ? "translateZ(30px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
          }}
        >
          <svg
            width="50"
            height="50"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
            style={{
              color: "var(--light-blue)",
              filter: dragActive
                ? "drop-shadow(0 2px 5px rgba(0,0,0,0.2))"
                : "none",
            }}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>
        <p
          className="mb-2"
          style={{
            transform: dragActive ? "translateZ(20px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
            fontWeight: dragActive ? "500" : "normal",
            color: "var(--primary-black)",
          }}
        >
          Drag and drop your image here, or click to select
        </p>
        <small
          className="text-muted d-block mb-4"
          style={{
            transform: dragActive ? "translateZ(15px)" : "translateZ(0)",
            transition: "transform 0.3s ease",
          }}
        >
          Supported formats: JPG, JPEG, PNG
        </small>

        <Form.Control
          ref={inputRef}
          type="file"
          style={{ display: "none" }}
          onChange={handleChange}
          accept=".jpg,.jpeg,.png"
        />

        <Button
          variant="primary"
          style={{
            borderRadius: "4px",
            padding: "8px 20px",
            fontSize: "16px",
            transform: dragActive ? "translateZ(25px)" : "translateZ(0)",
            transition: "transform 0.3s ease, box-shadow 0.3s ease",
            boxShadow: dragActive
              ? "0 8px 16px rgba(0,0,0,0.1)"
              : "0 4px 6px rgba(0,0,0,0.1)",
          }}
          onClick={(e) => {
            e.stopPropagation();
            onButtonClick();
          }}
        >
          Choose Image
        </Button>
      </div>
    </div>
  );
};

export default ImageUploader;
