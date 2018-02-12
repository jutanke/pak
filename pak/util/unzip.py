import zipfile
import tarfile
import subprocess
from pak import utils

def has_os_unzip():
    """ Checks if an OS tool for unzipping is installed

        returns True if so, o/w False
    """
    command = ["which", 'unzip']
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, err = process.communicate()
    return len(output) > 1  # path to binary OR empty


def unzip_using_os_tools(fzip, export_folder, verbose):
    """ unzips the given file to the given folder using os tools
    """
    #TODO: this has to be figured out eventually
    command = ["unzip", fzip, '-d', export_folder]
    utils.talk('\ttry unzip using OS tools', verbose)
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    utils.talk('\tunzip using OS tools', verbose)
    output, err = process.communicate()
    utils.talk('\terr:\t' + str(err), verbose)


def unzip(fzip, export_folder, verbose=False, force_os_tools=False):
    """
        @param fzip: {String} full path to the zip-file
        @param export_folder: {String} place to unzip
        @param verbose: {boolean} if true: debug
        @param force_os_tools: {boolean} if true: try to use OS tools
                                This is a horrible hack but it seems that
                                the python tools cannot unzip certain zip files
                                (that seem to be zipped on OSX) and I have no
                                time to find a 'good' solution so the ugly hack
                                is: use the system tools which for some reason
                                work! (on LINUX!!)
    """
    if force_os_tools:
        assert has_os_unzip()
        unzip_using_os_tools(fzip, export_folder, verbose)
    else:
        if fzip.endswith('.zip'):
            try:
                utils.talk("unzip " + fzip + " -> " + export_folder, verbose)
                zip_ref = zipfile.ZipFile(fzip, 'r')
                zip_ref.extractall(export_folder)
                zip_ref.close()
            except zipfile.BadZipFile as e:
                utils.talk('unzip exception:' + str(e), verbose)

                # -- try it the hard way -- use OS tools
                if has_os_unzip():
                    unzip_using_os_tools(fzip, export_folder, verbose)
                else:
                    raise

        elif fzip.endswith('tar.gz'):
            utils.talk("untar " + fzip + " -> " + export_folder, verbose)
            tar = tarfile.open(fzip, 'r:gz')
            tar.extractall(export_folder)
            tar.close()
