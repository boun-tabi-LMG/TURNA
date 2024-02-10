
TMP_CODE_ARCHIVE_NAME=${1:-tmp-turna.tar.gz}

pip install -U pip setuptools

git clone https://github.com/google-research/t5x.git
# ea66ec835a5b413ca9d211de96aa899900a84c13

wget -c https://github.com/google/flax/archive/refs/tags/v0.7.2.tar.gz

tar zxvf v0.7.2.tar.gz

pattern_to_replace=git\+https://github\.com/google/flax#egg=flax
replacement=file://localhost/${HOME}/flax-0.7.2#egg=flax

cd ~/t5x && sed -i.bak 's!'${pattern_to_replace}'!'${replacement}'!g' setup.py && cd -
cd ~/t5x && \
python3 -m pip install -e '.[gpu,tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
cd -

bash ./tpu_vm_setup_extract_the_code_archive.sh ${TMP_CODE_ARCHIVE_NAME}
