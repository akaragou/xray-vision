aws s3 cp ../checkpoints/ s3://yves/checkpoints/ --recursive
aws s3 cp ../results/ s3://yves/results/ --recursive
aws s3 cp ../summaries/ s3://yves/summaries/ --recursive
rm -r ../checkpoints/
rm -r ../results/
rm -r ../summaries/