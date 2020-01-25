
import argparse

import googleapiclient.discovery


def create_service():
    # Construct the service object for interacting with the Cloud Storage API -
    # the 'storage' service, at version 'v1'.
    # Authentication is provided by application default credentials.
    # When running locally, these are available after running
    # `gcloud auth application-default login`. When running on Compute
    # Engine, these are available from the environment.
    return googleapiclient.discovery.build('storage', 'v1')


def list_buckets(service, project_id):
    buckets = service.buckets().list(project=project_id).execute()
    return buckets


def main(project_id):
    service = create_service()
    buckets = list_buckets(service, project_id)
    print(buckets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud Project ID.')

    args = parser.parse_args()

    main(args.project_id)