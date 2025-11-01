FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN apk add --no-cache git \
    && npm install --production
COPY src ./src
COPY .env.example ./
CMD ["npm", "run", "start"]
