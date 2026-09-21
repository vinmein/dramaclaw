# AWS S3 media storage

Open **Settings → Media Storage → AWS S3**. Enter an existing bucket's name,
its AWS region (for example `ap-south-1`), an IAM access key ID and secret access
key, and a signed URL lifetime in seconds (default 1800; maximum 604800).
Click **Save AWS S3**. Saving stores configuration; it does not verify access by
uploading a file. Blank credential fields retain the saved credentials.

The backend uploads reference images, video and audio under `relay/` and gives
model providers temporary signed GET URLs. Keep S3 Block Public Access enabled;
no public bucket policy or public-read ACL is needed. This configures the existing
media relay, not a backup or migration of local project files.

Grant the IAM identity the following policy, replacing `YOUR_BUCKET`:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:PutObject", "s3:GetObject"],
    "Resource": "arn:aws:s3:::YOUR_BUCKET/relay/*"
  }]
}
```

Buckets using a customer-managed KMS key also need the appropriate KMS key
permissions. Choose a URL lifetime long enough for the model provider to fetch
references. URL expiry does not delete objects; configure an S3 lifecycle rule
for the `relay/` prefix to remove transient uploads after your chosen retention
period. S3 storage and transfer charges apply.

Credentials are saved server-side in the existing local settings database, with
masked previews returned to the browser. S3 credential inputs are held only in
component memory and cleared after saving. Protect and back up the settings
database accordingly. This form supports long-lived IAM access keys; temporary
STS credentials with session tokens are not supported.

For environment-based setup before saving settings in the UI:

```dotenv
MEDIA_RELAY_PROVIDER=aws_s3
MEDIA_RELAY_TTL_SECONDS=1800
S3_RELAY_REGION=ap-south-1
S3_RELAY_BUCKET=YOUR_BUCKET
S3_RELAY_AK=YOUR_ACCESS_KEY_ID
S3_RELAY_SK=YOUR_SECRET_ACCESS_KEY
```

Restart the backend after changing environment variables. Saved database
configuration takes precedence. The S3 implementation uses the AWS SDK and
Signature Version 4: [AWS presigned URL documentation](https://docs.aws.amazon.com/boto3/latest/guide/s3-presigned-urls.html).
