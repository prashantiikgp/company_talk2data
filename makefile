📄 Makefile (place this in your Netflix_Project/ root)
makefile
Copy
Edit
# --------------------------------------------
# 📦 Qdrant Docker Compose Commands
# --------------------------------------------

DC := docker compose
COMPOSE_FILE := docker-compose.yml
SERVICE := qdrant_server

## 🟢 Start Qdrant in detached mode
up:
	@echo "🚀 Starting Qdrant Server..."
	$(DC) up -d

## 🔴 Stop Qdrant container
stop:
	@echo "🛑 Stopping Qdrant Server..."
	$(DC) stop

## 🔁 Restart the Qdrant container
restart:
	@echo "♻️  Restarting Qdrant Server..."
	$(DC) restart

## 💣 Kill and remove all Qdrant containers + volumes
kill:
	@echo "💥 Killing & Removing Qdrant containers and volumes..."
	$(DC) down -v --remove-orphans

## 📄 Tail logs from Qdrant
logs:
	@echo "📜 Streaming logs..."
	$(DC) logs -f $(SERVICE)

## 🧹 Clean up orphan containers and networks
clean:
	@echo "🧹 Cleaning up unused containers and networks..."
	docker system prune -f

## 🔍 Show running containers
ps:
	$(DC) ps

## 🧱 Rebuild Qdrant with fresh volumes
rebuild:
	@echo "🔨 Rebuilding Qdrant server and volumes..."
	$(DC) down -v
	$(DC) up -d --build

.PHONY: up stop restart kill logs clean ps rebuild
✅ How to use:
In terminal:

bash
Copy
Edit
make up         # Start the server
make stop       # Stop the container
make logs       # See logs
make kill       # Remove everything
make rebuild    # Fully wipe and rebuild
make ps         # See status