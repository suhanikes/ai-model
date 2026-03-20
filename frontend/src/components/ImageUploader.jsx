import React from "react";

export function ImageUploader({ onImageSelected }) {
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    onImageSelected({ file, url });
  };

  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm font-medium text-slate-200">
        Upload garment photo
      </label>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="block w-full text-sm text-slate-200 file:mr-4 file:rounded-md file:border-0 file:bg-indigo-600 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-indigo-500"
      />
      <p className="text-xs text-slate-400">
        Use clear, well-lit product images up to 10MB.
      </p>
    </div>
  );
}

