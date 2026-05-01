import flwr as fl

if __name__ == "__main__":
    print("🚀 Starting FL server...")

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),  # keep 10 for testing
    )