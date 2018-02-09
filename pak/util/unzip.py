import zipfile
import tarfile
import subprocess
from pak import utils

def unzip(fzip, export_folder, verbose=False):
    """
        @param fzip: {String} full path to the zip-file
    """
    if fzip.endswith('.zip'):
        try:
            utils.talk("unzip " + fzip + " -> " + export_folder, verbose)
            zip_ref = zipfile.ZipFile(fzip, 'r')
            zip_ref.extractall(export_folder)
            zip_ref.close()
        except zipfile.BadZipFile as e:
            utils.talk('unzip exception:' + str(e), verbose)

            # -- try it the hard way -- use OS tools
            try:
                command = ["unzip", fzip, '-d', export_folder]
                utils.talk('\ttry unzip using OS tools', verbose)
                process = subprocess.Popen(command, stdout=subprocess.PIPE)
                utils.talk('\tunzip using OS tools', verbose)
                output, err = process.communicate()
                #utils.talk('\toutput:\t' + str(output), verbose)
                utils.talk('\terr:\t' + str(err), verbose)

            except FileNotFoundError as fe:  # no unzip-tool found
                raise e  # bubble-up the original error

    elif fzip.endswith('tar.gz'):
        utils.talk("untar " + fzip + " -> " + export_folder, verbose)
        tar = tarfile.open(fzip, 'r:gz')
        tar.extractall(export_folder)
        tar.close()
