import React, { useState, useEffect } from "react";
import {
  Card,
  Form,
  Button,
  Alert,
  InputGroup,
  Row,
  Col,
  Spinner,
} from "react-bootstrap";
import axios from "axios";

function Settings() {
  const [objectDistance, setObjectDistance] = useState(300);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [initialLoad, setInitialLoad] = useState(true);

  // Fetch current settings when component mounts
  useEffect(() => {
    const fetchSettings = async () => {
      setLoading(true);
      try {
        const response = await axios.get("/settings");
        if (response.data) {
          if (response.data.object_distance) {
            setObjectDistance(response.data.object_distance);
          }
        }
      } catch (err) {
        console.error("Error fetching settings:", err);
        setError("Could not load settings");
      } finally {
        setLoading(false);
        setInitialLoad(false);
      }
    };

    fetchSettings();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess(false);

    // Validate object distance
    const distance = parseFloat(objectDistance);
    if (isNaN(distance) || distance <= 0) {
      setError("Object distance must be a positive number");
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post("/settings", {
        object_distance: distance,
      });

      if (response.data.success) {
        setSuccess(true);
        // Hide success message after 3 seconds
        setTimeout(() => setSuccess(false), 3000);
      } else {
        setError("Failed to update settings");
      }
    } catch (err) {
      console.error("Error updating settings:", err);
      setError("Could not update settings. Server error.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="shadow-sm">
      <Card.Header as="h5" className="bg-light">
        <div className="d-flex align-items-center">
          <div className="me-3">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              fill="currentColor"
              className="bi bi-gear"
              viewBox="0 0 16 16"
            >
              <path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z" />
              <path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115l.094-.319z" />
            </svg>
          </div>
          <div>Configuration Settings</div>
        </div>
      </Card.Header>
      <Card.Body>
        {initialLoad ? (
          <div className="text-center py-3">
            <Spinner animation="border" role="status" variant="primary">
              <span className="visually-hidden">Loading...</span>
            </Spinner>
            <p className="mt-2">Loading settings...</p>
          </div>
        ) : (
          <Form onSubmit={handleSubmit}>
            {error && (
              <Alert variant="danger" dismissible onClose={() => setError("")}>
                {error}
              </Alert>
            )}

            {success && (
              <Alert
                variant="success"
                dismissible
                onClose={() => setSuccess(false)}
              >
                Settings saved successfully!
              </Alert>
            )}

            <Form.Group className="mb-3">
              <Form.Label>
                <strong>Object Distance (mm)</strong>
              </Form.Label>
              <InputGroup>
                <Form.Control
                  type="number"
                  placeholder="Enter distance from camera to object"
                  value={objectDistance}
                  onChange={(e) => setObjectDistance(e.target.value)}
                  min="1"
                />
                <InputGroup.Text>mm</InputGroup.Text>
              </InputGroup>
              <Form.Text className="text-muted">
                This affects size measurement accuracy. Default: 300mm
              </Form.Text>
            </Form.Group>

            <div className="d-flex justify-content-end">
              <Button
                variant="primary"
                type="submit"
                disabled={loading}
                className="d-flex align-items-center"
              >
                {loading && (
                  <Spinner
                    as="span"
                    animation="border"
                    size="sm"
                    role="status"
                    aria-hidden="true"
                    className="me-2"
                  />
                )}
                Save Settings
              </Button>
            </div>
          </Form>
        )}
      </Card.Body>
    </Card>
  );
}

export default Settings;
