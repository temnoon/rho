/**
 * API utilities for communicating with the Rho backend.
 * 
 * Provides standardized functions for making API calls with proper error handling.
 */

// Use environment variable or fallback to default
const API = import.meta.env.VITE_API_URL || "http://localhost:8192";

export function apiUrl(path) {
  if (!path) return API;
  const base = API.replace(/\/+$/, "");
  const suffix = path.startsWith("/") ? path.replace(/^\/+/, "") : path;
  return `${base}/${suffix}`;
}

export async function safeFetch(path, opts = {}) {
  const url = apiUrl(path);
  try {
    const res = await fetch(url, opts);
    if (!res.ok) {
      let bodyText = "<no body>";
      try {
        bodyText = await res.text();
      } catch (e) {
        bodyText = `<unable to read body: ${String(e)}>`;
      }
      const err = new Error(`${url} returned ${res.status} ${res.statusText} - ${bodyText}`);
      err.status = res.status;
      throw err;
    }
    return res;
  } catch (err) {
    console.error("[RHO] Network error for", url, err);
    throw err;
  }
}

export function formatNumber(n, digits = 4) {
  if (!Number.isFinite(n)) return "NaN";
  return Number(n).toFixed(digits);
}