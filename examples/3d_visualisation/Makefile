all: examples

clean: 
	@echo "cleaning up"
	@rm -f *.png *.pyc

examples:
	@for f in `/bin/ls *.py 2> /dev/null` ; do python $$f ; done
