# Contributions

## Releasing a new version

1. Update version.py to the to-be-released version, say 1.2.3
2. Commit and push the changes to version.py to master
3. Tag the commit with v1.2.3 to match the version in version.py
4. Push the tag
5. Create a Release in github from the new tag. This will trigger the "Upload Python Package" workflow.