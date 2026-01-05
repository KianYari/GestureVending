push:
	docker buildx build --platform linux/amd64 -t ghcr.io/kianyari/dari-server:latest .
	docker push ghcr.io/kianyari/dari-server:latest

local:
	docker compose down
	docker compose up -d