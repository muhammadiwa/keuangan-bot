FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN apk add --no-cache git \
    && npm install
COPY src ./src
COPY .env ./
CMD ["npm", "run", "dev"]
