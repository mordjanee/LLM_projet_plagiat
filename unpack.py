import os
import pandas as pd
import sys
import shutil
import argparse
import json

def organize_plagiarism_data(input_dir, output_dir):

    # Définir les chemins des sous-dossiers 'source' et 'suspicous' dans le répertoire de sortie
    output_origine = os.path.join(output_dir, "source")
    output_suspicieux = os.path.join(output_dir, "suspicious")

    os.makedirs(output_origine, exist_ok=True)
    os.makedirs(output_suspicieux, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            if dir_name.lower() == "source":
                source_dir = os.path.join(root, dir_name)
                copy_files(source_dir, output_origine)
            elif dir_name.lower() == "suspicious":
                source_dir = os.path.join(root, dir_name)
                copy_files(source_dir, output_suspicieux)

    print(f"Organisation terminée. Les fichiers sont dans le dossier : {output_dir}")


def copy_files(source_dir, destination_dir):
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        if os.path.isfile(source_file):
            destination_file = os.path.join(destination_dir, file_name)
            shutil.copy2(source_file, destination_file) 
            print(f"Copié : {source_file} -> {destination_file}")


def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)


if __name__ == "__main__":

    sys.argv = [arg for arg in sys.argv if not arg.startswith("--f=")]

    parser = argparse.ArgumentParser(description="Organise les fichiers de plagiat")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de configuration JSON")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        input_dir = config.get("input_dir")
        output_dir = config.get("output_dir")

        if not input_dir or not output_dir:
            raise ValueError("Les clés 'input_dir' et 'output_dir' doivent être spécifiées dans le fichier config.json.")

        organize_plagiarism_data(input_dir, output_dir)

    except FileNotFoundError:
        print(f"Erreur : Le fichier de configuration '{args.config}' est introuvable.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier de configuration '{args.config}' n'est pas un fichier JSON valide.")
        exit(1)
    except ValueError as e:
        print(f"Erreur : {e}")
        exit(1)



#python unpack.py --config=config.json
