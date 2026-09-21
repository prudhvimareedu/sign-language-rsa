const API_URL =
  "https://sign-language-rsa.onrender.com";

export async function predictImage(blob) {
  const formData = new FormData();

  formData.append(
    "file",
    blob,
    "webcam.jpg"
  );

  const response = await fetch(
    `${API_URL}/predict`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error(
      `Prediction request failed: ${response.status}`
    );
  }

  return response.json();
}