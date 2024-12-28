#!make

include .env
export

clean:
	rm -rf ./data/external
	rm -rf ./data/processed
	rm -rf ./models
	rm -rf ./*.log

download: venv
	pip install kaggle
	kaggle competitions download -c playground-series-s4e10 -p data/external

include Makefile.venv
Makefile.venv:
	curl \
		-o Makefile.fetched \
		-L "https://github.com/sio/Makefile.venv/raw/v2023.04.17/Makefile.venv"
	echo "fb48375ed1fd19e41e0cdcf51a4a0c6d1010dfe03b672ffc4c26a91878544f82 *Makefile.fetched" \
		| sha256sum --check - \
		&& mv Makefile.fetched Makefile.venv
