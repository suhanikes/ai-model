import axios from "axios";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:4000/api";

export async function uploadImage(file) {
  const formData = new FormData();
  formData.append("image", file);
  // Do not set Content-Type; let the browser set it with the correct boundary
  const res = await axios.post(`${API_BASE}/upload`, formData);
  return res.data;
}

export async function lassoSegmentation({
  imageId,
  lassoPoints,
  selectedColor,
}) {
  if (!imageId || !Array.isArray(lassoPoints) || lassoPoints.length < 3) {
    throw new Error("Need imageId and at least 3 lasso points.");
  }
  const res = await axios.post(
    `${API_BASE}/lasso-segmentation`,
    {
      imageId,
      lasso_points: lassoPoints,
      selected_color: selectedColor ?? "#ff3366",
    },
    { headers: { "Content-Type": "application/json" } }
  );
  return res.data;
}

