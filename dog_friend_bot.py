import os
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

global model

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


def startCommand(update: Update, context: CallbackContext) -> None:
    """Sending a message when the command /start is issued."""
    update.message.reply_text('Hi! Send me a photo of a dog and I\'ll try to guess its breed')


def helpCommand(update: Update, context: CallbackContext) -> None:
    """Sending a message when the command /help is issued."""
    update.message.reply_text('Send me a photo of a dog and I\'ll try to guess its breed')


def getmodel():
    model = models.resnet50()
    for param in model.parameters():
        param.requiers_grad = False
    model.fc = nn.Linear(2048, 120)

    path = 'model_weights100ep.pth'
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


model = getmodel()


def predictBreed(photo_file):
    global model
    image = Image.open(photo_file).convert('RGB')
    classes = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
               'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
               'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
               'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
               'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
               'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard',
               'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever',
               'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie',
               'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
               'doberman', 'english_foxhound', 'english_setter', 'english_springer',
               'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog',
               'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer',
               'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog',
               'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel',
               'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond',
               'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever',
               'lakeland_terrier', 'leonberg', 'lhasa', 'malamute',
               'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher',
               'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound',
               'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon',
               'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone',
               'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed',
               'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
               'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
               'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer',
               'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle',
               'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
               'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']

    transforms_test = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    image_tensor = transforms_test(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    output = model(image_tensor)
    index = output.argmax().item()
    print(index)
    return classes[index].replace("_", " ")


def photo(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    photo_file = context.bot.getFile(update.message.photo[-1].file_id)
    photo_file.download("user_photo.jpg")
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')

    update.message.reply_text('Such a cuttie! Please wait a second!')
    update.message.reply_markdown('This is probably ' + str(predictBreed('user_photo.jpg')))


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    TOKEN = "bot private token"
    updater = Updater(TOKEN)  # , use_context=True)

    PORT = int(os.environ.get('PORT', '8443'))

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", startCommand))
    dispatcher.add_handler(CommandHandler("help", helpCommand))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo))
    updater.start_polling(drop_pending_updates=True)

    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN,
                          webhook_url='https://yourherokuappname.herokuapp.com/' + TOKEN)
    updater.idle()


if __name__ == '__main__':
    main()
