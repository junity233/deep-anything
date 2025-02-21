from deepanything.Server.Server import DeepAnythingServer
import argparse
import json
def main():
    parser = argparse.ArgumentParser(prog="deepanything",description="Run a DeepAnything Server.")
    parser.add_argument('--host', type=str, required=False, help='Specific the host to listen.If specified,the host will be overwritten by this.')
    parser.add_argument('--port', type=int, required=False, help='Specific the port to listen.If specified,the port will be overwritten by this.')
    parser.add_argument('--config', type=str, required=True, help='Specific the confi path.')

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        server = DeepAnythingServer(host=args.host,port=args.port,config_object=config)
        server.run()
    else:
        print("No config file specified.")

if __name__ == "__main__":
    main()