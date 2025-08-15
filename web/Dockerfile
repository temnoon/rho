# Multi-stage Dockerfile for the Rho web frontend (Vite + React) + nginx static server
#
# Stage 1: Build the production static assets using node (Vite)
# Stage 2: Serve the built assets with nginx:alpine
#
# This Dockerfile is written to be context-relative so it works when the
# build context is the `web/` directory (i.e., `docker build -f web/Dockerfile web`).
#
# Notes:
#  - To inject an API base at build time, set the build-arg VITE_API_BASE.
#    In your app, reference it as import.meta.env.VITE_API_BASE.
#  - For local development, run `npm run dev` from the web/ folder instead of using this image.
#
# Minimal, secure, reasonably small image for a demo site.

###########
# Builder #
###########
FROM node:20-alpine AS builder

# Install dependencies required for some packages
RUN apk add --no-cache git python3 make g++ bash

WORKDIR /app

# Copy package manifests first to leverage docker layer caching
# Context-relative: expects package.json etc to be in the build context root (./web when using context: ./web)
COPY package.json package-lock.json* yarn.lock* ./

# Default build arg for API base (used by Vite as VITE_API_BASE)
ARG VITE_API_BASE="/api"
ENV VITE_API_BASE=${VITE_API_BASE}

# Install deps
RUN if [ -f package-lock.json ]; then npm ci; else npm install --no-audit --no-fund; fi

# Copy source files from the context root (e.g., ./web)
COPY . .

# Build production assets
RUN npm run build

#########
# Final #
#########
FROM nginx:alpine

# Expose port (optional; docker run -p needed)
EXPOSE 80

# If you need to provide a custom nginx config from outside the web context,
# mount it at runtime or adjust the build to include it. For the context-relative
# web build we do not COPY from ../deploy; the default nginx config will be used.
# Place a custom nginx.conf in the web/ build context to have it included in the image.
# COPY from the build context root (./web/nginx.conf) into the nginx config path:
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built static files from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy a simple static healthz if desired
RUN echo "ok" > /usr/share/nginx/html/healthz

# (Optional) set a non-root user for runtime if desired - keep nginx defaults for simplicity
# Ensure file permissions are correct
RUN chown -R nginx:nginx /usr/share/nginx/html

# Keep container running with nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
