import os
import datetime
import shutil
import glob




def make_log_dir(config):
    log_dir = os.path.join('.', 'logs', 'search-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    os.mkdir(os.path.join(log_dir, 'scripts'))

    for script in glob.glob('*.py'):
        dst_file = os.path.join(log_dir, 'scripts', os.path.basename(script))
        shutil.copyfile(script, dst_file)

    os.mkdir(os.path.join(log_dir, 'scripts','gnas'))
    for script in glob.glob('./gnas/*.py'):
        dst_file = os.path.join(log_dir, 'scripts','gnas', os.path.basename(script))
        shutil.copyfile(script, dst_file)

    os.mkdir(os.path.join(log_dir, 'scripts', 'gnas','common'))
    for script in glob.glob('./gnas/common/*.py'):
        dst_file = os.path.join(log_dir, 'scripts', 'gnas','common', os.path.basename(script))
        shutil.copyfile(script, dst_file)

    os.mkdir(os.path.join(log_dir, 'scripts', 'gnas', 'genetic_algorithm'))
    for script in glob.glob('./gnas/genetic_algorithm/*.py'):
        dst_file = os.path.join(log_dir, 'scripts', 'gnas', 'genetic_algorithm', os.path.basename(script))
        shutil.copyfile(script, dst_file)

    os.mkdir(os.path.join(log_dir, 'scripts', 'gnas', 'modules'))
    for script in glob.glob('./gnas/modules/*.py'):
        dst_file = os.path.join(log_dir, 'scripts', 'gnas', 'modules', os.path.basename(script))
        shutil.copyfile(script, dst_file)

    os.mkdir(os.path.join(log_dir, 'scripts', 'gnas', 'search_space'))
    for script in glob.glob('./gnas/search_space/*.py'):
        dst_file = os.path.join(log_dir, 'scripts', 'gnas', 'search_space', os.path.basename(script))
        shutil.copyfile(script, dst_file)

    os.mkdir(os.path.join(log_dir, 'scripts', 'models'))
    for script in glob.glob('./models/*.py'):
        dst_file = os.path.join(log_dir, 'scripts', 'models', os.path.basename(script))
        shutil.copyfile(script, dst_file)

    os.mkdir(os.path.join(log_dir, 'scripts', 'modules'))
    for script in glob.glob('./modules/*.py'):
        dst_file = os.path.join(log_dir, 'scripts', 'modules', os.path.basename(script))
        shutil.copyfile(script, dst_file)
    return log_dir



