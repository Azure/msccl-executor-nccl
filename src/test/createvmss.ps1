param(
    [string]$Subscription = "hpcperf1",
	[string]$VmssName = "vmss-msccl-dev",
    [string]$ResourceGroup = "msccl-dev",
	[string]$Location = "South Central US",
	[string]$ComputerNamePrefix = "Msccl-Dev-",
	[string]$VmSku = "Standard_ND96amsr_A100_v4",
	[string]$InstanceCount = 2,
	[string]$Image = "microsoft-dsvm:ubuntu-hpc:2004:latest",
	[string]$DiskSize = 60,
	[string]$VnetName = "vnet-msccl-dev",
	[string]$NsgName = "basicNsgmsccl-dev-vmss-nic01",
	[string]$IpName = "ip-msccl-dev",
	[string]$AdminName = "azureuser",
	[string]$SshPublicKey = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCi2fAdE8Limtvt/tP4FpwNr6hTsJcb0Viaxw9mPdLiYeYBCn8Ar9MWcdF7djdA6CTrSvl1RZQNerlq17Go4WxyTG4GRr6/ZDv2cd6Y1Cs/xD+VX3V6pFK1E2K1cZAs9wEF/Y1xsdtNpHsScp9e73MYKBeDExTgO7WE9EkhwLXQXJax5CozWjU6Z9F+0lFbrrvkvcbR3hdiNWDksK2aVKchtxzoqiQVe9tGTNdCiKaqG8iPRi3XAVZ0p7AG0nVI3YjYQ1CVln7RQBxDz4Lhht7XxVGAZWtEuCNd9mJ8P60f+4QWBx763wM1iOq64FpZrpprUHoZ6gKZP3ZpOoUSWLk7hPiiaJVM/bQjy/9ZVstM+0JcJ9X/NXgEiQimRQPs43WTPWv4OYlrWYjt15QBDTFhUdS1k4yHwnfsXXBCqp520E4ann6kAsXZ0cWULkTVa9EqQsfWkCBuz/F6T4XKFrX6WpsU7LuNfOQxY0d4C/JmVDFH84QNEOiXbggdjcXelFU= fareast\liand@tesla"
)

az login --use-device-code
az account set --subscription $Subscription

az vmss create --name $VmssName --resource-group $ResourceGroup --location $Location --computer-name-prefix $ComputerNamePrefix --vm-sku $VmSku --instance-count $InstanceCount --image $Image --os-disk-size-gb $DiskSize --priority Regular --single-placement-group true --platform-fault-domain-count 1 --accelerated-networking false --vnet-name $VnetName --nsg $NsgName --subnet default --public-ip-address $IpName --authentication-type ssh --admin-username $AdminName --ssh-key-values $SshPublicKey --lb-sku Standard