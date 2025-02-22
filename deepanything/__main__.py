from deepanything.Server.Server import DeepAnythingServer
import argparse
import json
from .metadatas import VERSION
def main():
    parser = argparse.ArgumentParser(prog=f"deepanything {VERSION}",description="Run a DeepAnything Server.")
    parser.add_argument('--host', type=str, required=False, help='Specific the host to listen.If specified,the host will be overwritten by this.')
    parser.add_argument('--port', type=int, required=False, help='Specific the port to listen.If specified,the port will be overwritten by this.')
    parser.add_argument('--config', type=str, required=True, help='Specific the confi path.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, encoding='utf-8') as f:
            config = json.load(f)
        server = DeepAnythingServer(host=args.host, port=args.port, config=config)
        server.run()
    else:
        print("No config file specified.")

if __name__ == "__main__":
    main()