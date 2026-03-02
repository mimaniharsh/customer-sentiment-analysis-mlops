name: CD — Deploy to Production

# Only run this manually when you have ECS/ECR set up.
# For running the platform, use the ML workflow instead.
on:
  workflow_dispatch:
    inputs:
      environment:
        description: "Target environment"
        required: true
        default: staging
        type: choice
        options: [staging, production]

jobs:
  deploy:
    name: Deploy — ${{ github.event.inputs.environment }}
    runs-on: ubuntu-latest
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      AWS_DEFAULT_REGION: ${{ secrets.AWS_REGION }}
    steps:
      - uses: actions/checkout@v4

      - name: Login to Amazon ECR
        run: |
          aws ecr get-login-password --region ${{ secrets.AWS_REGION }} | \
          docker login --username AWS --password-stdin ${{ secrets.ECR_REGISTRY }}

      - name: Build and push API image
        run: |
          docker build -t ${{ secrets.ECR_REGISTRY }}/sentient-api:${{ github.sha }} apps/api/
          docker push ${{ secrets.ECR_REGISTRY }}/sentient-api:${{ github.sha }}

      - name: Build and push Web image
        run: |
          docker build \
            --build-arg NEXT_PUBLIC_API_URL=${{ secrets.API_PUBLIC_URL }} \
            -t ${{ secrets.ECR_REGISTRY }}/sentient-web:${{ github.sha }} apps/web/
          docker push ${{ secrets.ECR_REGISTRY }}/sentient-web:${{ github.sha }}

      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster ${{ secrets.ECS_CLUSTER }} --service sentient-api --force-new-deployment
          aws ecs update-service --cluster ${{ secrets.ECS_CLUSTER }} --service sentient-web --force-new-deployment
