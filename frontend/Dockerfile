FROM node:20-alpine

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

# --host makes the Vite dev server reachable from outside the container
# (i.e. from the port mapping in docker-compose.yml).
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
