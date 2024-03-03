#!make
include .env

.PHONY: build
build:
	docker build --target $(target)

.PHONY: run
test:
	docker run \

.PHONY: test
test:
	docker build --target $(target)